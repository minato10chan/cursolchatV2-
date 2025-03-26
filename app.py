# SQLiteをpysqlite3で上書き
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Successfully overrode sqlite3 with pysqlite3")
except ImportError:
    print("Failed to override sqlite3 with pysqlite3")

import streamlit as st
import datetime

# 最初のStreamlitコマンドとしてページ設定を行う
st.set_page_config(page_title='🦜🔗 Ask the Doc App', layout="wide")

# セッション状態の初期化
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

from langchain_openai import OpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
# --- LLM --- (componentsフォルダにllm.pyを配置する)---
from components.llm import llm
from components.llm import oai_embeddings
# --- LLM ---
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
import pandas as pd
# ChromaDBとVectorStoreのインポートは後で行う (SQLite修正後)
import io

# カテゴリの定義
MAJOR_CATEGORIES = [
    "1. 物件概要",
    "2. 地域特性・街のプロフィール",
    "3. 教育・子育て",
    "4. 交通・アクセス",
    "5. 安全・防災",
    "6. 行政施策・政策",
    "7. 生活利便性",
    "8. 不動産市場",
    "9. 地域コミュニティ",
    "10. その他（個別の懸念・特殊事情）"
]

# 中カテゴリの定義
MEDIUM_CATEGORIES = {
    "1. 物件概要": [
        "1.1 完成時期",
        "1.2 販売開始",
        "1.3 建築確認",
        "1.4 間取り・仕様",
        "1.5 設備・オプション",
        "1.6 デザイン・外観",
        "1.7 建築会社・デベロッパー",
        "1.8 価格・費用",
        "1.9 資産価値・売却",
        "1.10 契約・手続き"
    ],
    "2. 地域特性・街のプロフィール": [
        "2.1 概要・エリア区分",
        "2.2 人口・居住特性",
        "2.3 街の歴史・地域史",
        "2.4 地理的特性",
        "2.5 自然環境",
        "2.6 地域イベント・伝統行事",
        "2.7 都市連携・姉妹都市情報",
        "2.8 治安・騒音・環境整備",
        "2.9 風景・景観・街並み",
        "2.10 観光・ 地元特産品・名産・グルメ"
    ],
    "3. 教育・子育て": [
        "3.1 保育園・幼稚園",
        "3.2 小学校・中学校",
        "3.3 学童・放課後支援",
        "3.4 習い事・塾",
        "3.5 子育て支援制度",
        "3.6 公園・遊び場",
        "3.7 病院・小児科",
        "3.8 学区・教育水準",
        "3.9 待機児童・入園状況",
        "3.10 学校イベント・行事"
    ],
    "4. 交通・アクセス": [
        "4.1 最寄り駅・路線",
        "4.2 電車の混雑状況",
        "4.3 バス路線・本数",
        "4.4 駐輪場・駐車場",
        "4.5 道路交通量・渋滞",
        "4.6 車移動のしやすさ",
        "4.7 通勤・通学時間",
        "4.8 高速道路・インター",
        "4.9 タクシー・ライドシェア",
        "4.10 空港・新幹線アクセス"
    ],
    "5. 安全・防災": [
        "5.1 防犯カメラ・交番の有無",
        "5.2 避難場所・防災拠点",
        "5.3 ハザードマップ（洪水・地震）",
        "5.4 土砂災害リスク",
        "5.5 耐震性・建物強度",
        "5.6 火災リスク・消防体制",
        "5.7 夜道の安全性",
        "5.8 台風・風害・雪害対策",
        "5.9 地震・液状化リスク",
        "5.10 交通事故・子ども見守り"
    ],
    "6. 行政施策・政策": [
        "6.1 市政・行政組織",
        "6.2 再開発・都市計画",
        "6.3 交通インフラ整備",
        "6.4 公共施設運営・市民サービス",
        "6.5 ゴミ収集・清掃環境",
        "6.6 商業・産業振興策",
        "6.7 住宅政策・住環境整備",
        "6.8 福祉・医療支援",
        "6.9 補助金・助成制度",
        "6.10 行政評価・市民参加"
    ],
    "7. 生活利便性": [
        "7.1 スーパー・買い物環境",
        "7.2 コンビニ・ドラッグストア",
        "7.3 銀行・金融機関・郵便局",
        "7.4 公共施設",
        "7.5 病院・クリニック・夜間救急",
        "7.6 文化施設・美術館・劇場",
        "7.7 スポーツ施設・ジム",
        "7.8 娯楽施設・カラオケ・映画館",
        "7.9 飲食店・グルメスポット",
        "7.10 宅配サービス・ネットスーパー"
    ],
    "8. 不動産市場": [
        "8.1 地価の変動・推移",
        "8.2 将来の売却しやすさ",
        "8.3 賃貸需要・投資価値",
        "8.4 住宅ローン・金利動向",
        "8.5 人気エリアの傾向",
        "8.6 空き家・中古市場動向",
        "8.7 固定資産税・税制優遇",
        "8.8 マンション vs 戸建て",
        "8.9 住み替え・転勤時の影響",
        "8.10 不動産会社の評判・実績"
    ],
    "9. 地域コミュニティ": [
        "9.1 自治会・町内会の活動",
        "9.2 地域の祭り",
        "9.3 ボランティア活動",
        "9.4 住民の意識・口コミ",
        "9.5 高齢者支援・福祉施設",
        "9.6 外国人コミュニティ",
        "9.7 風評・悪評の実態",
        "9.8 市民幸福度・満足度",
        "9.9 地域振興・コミュニティ活性化"
    ],
    "10. その他（個別の懸念・特殊事情）": [
        "10.1 スマートシティ・DX施策",
        "10.2 エコ対策・再生可能エネルギー",
        "10.3 観光客・短期滞在者の影響",
        "10.4 景観条例・建築規制",
        "10.5 駐車場問題・車両ルール",
        "10.6 宅配便・物流インフラ"
    ]
}

# カスタムRAGプロンプトテンプレートを定義
# langchainhubに依存せずに自前でプロンプトを定義
RAG_PROMPT_TEMPLATE = """あなたは不動産会社の営業担当です。取り扱っている物件を中心をしたエリアに対して、エリアの魅力や特徴、生活環境について詳しく説明することが得意です。以下の情報源を元に、質問に対して具体的で魅力的な回答を提供してください。

情報源:
{context}

質問: {question}

回答の際は以下のポイントを意識してください：
1. エリアの魅力や特徴を具体的に伝える
2. 実際に住むイメージが湧くような説明をする
3. 交通、教育、商業施設、医療、公共施設などの生活利便性について触れる
4. 数字やデータを用いて客観的な情報も提供する
5. 物件の見学意欲が高まるような表現を使う
6. 情報源に記載がない内容については「この点については情報がありません」と正直に伝える

回答:"""

# グローバル変数の初期化
vector_store = None
vector_store_available = False

# VectorStoreのインスタンスを初期化する関数
def initialize_vector_store():
    global vector_store, vector_store_available
    if vector_store is not None:
        return vector_store
        
    try:
        from src.vector_store import VectorStore
        vector_store = VectorStore()
        vector_store_available = True
        print("VectorStore successfully initialized")
        return vector_store
    except Exception as e:
        vector_store_available = False
        print(f"Error initializing VectorStore: {e}")
        return None

# 初期化を試みる
initialize_vector_store()

def register_document(uploaded_file, additional_metadata=None):
    """
    アップロードされたファイルをChromaDBに登録する関数。
    additional_metadata: 追加のメタデータ辞書
    """
    if not vector_store_available:
        st.error("データベース接続でエラーが発生しました。ChromaDBが使用できません。")
        return
    
    if uploaded_file is not None:
        try:
            # ファイルの内容を読み込み - 複数のエンコーディングを試す
            content = None
            encodings_to_try = ['utf-8', 'shift_jis', 'cp932', 'euc_jp', 'iso2022_jp']
            
            file_bytes = uploaded_file.getvalue()
            
            # 異なるエンコーディングを試す
            for encoding in encodings_to_try:
                try:
                    content = file_bytes.decode(encoding)
                    st.success(f"ファイルを {encoding} エンコーディングで読み込みました")
                    break
                except UnicodeDecodeError:
                    continue
            
            # どのエンコーディングでも読み込めなかった場合
            if content is None:
                st.error("ファイルのエンコーディングを検出できませんでした。UTF-8, Shift-JIS, EUC-JP, ISO-2022-JPのいずれかで保存されたファイルをお試しください。")
                return
            
            # メモリ内でテキストを分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=10,
                add_start_index=True,
                separators=["\n\n", "\n", "。", ".", " ", ""],
            )
            
            # 基本メタデータの作成
            base_metadata = {'source': uploaded_file.name}
            
            # 追加メタデータが指定されている場合は統合
            if additional_metadata:
                base_metadata.update(additional_metadata)
            
            # ドキュメントを作成
            from langchain_core.documents import Document
            raw_document = Document(
                page_content=content,
                metadata=base_metadata
            )
            
            # ドキュメントを分割
            documents = text_splitter.split_documents([raw_document])

            # セッション状態にドキュメントを保存
            st.session_state.documents.extend(documents)

            # IDsの作成
            original_ids = []
            for i, doc in enumerate(documents):
                source_ = os.path.splitext(uploaded_file.name)[0]  # 拡張子を除く
                start_ = doc.metadata.get('start_index', i)
                id_str = f"{source_}_{start_:08}" #0パディングして8桁に
                original_ids.append(id_str)

            # グローバルのVectorStoreインスタンスを使用
            global vector_store
            
            # ドキュメントの追加（UPSERT）
            vector_store.upsert_documents(documents=documents, ids=original_ids)

            st.success(f"{uploaded_file.name} をデータベースに登録しました。")
            st.info(f"{len(documents)}件のチャンクに分割されました")
        except Exception as e:
            st.error(f"ドキュメントの登録中にエラーが発生しました: {e}")
            st.error("エラーの詳細:")
            st.exception(e)

def manage_chromadb():
    """
    ChromaDBを管理するページの関数。
    """
    st.header("ChromaDB 管理")

    if not vector_store_available:
        st.error("ChromaDBの接続でエラーが発生しました。現在、ベクトルデータベースは使用できません。")
        st.warning("これはSQLiteのバージョンの非互換性によるものです。Streamlit Cloudでの実行には制限があります。")
        return

    # グローバルのvector_storeを使用
    global vector_store

    # 1.ドキュメント登録
    st.subheader("ドキュメントをデータベースに登録")
    
    # ファイルアップロード
    uploaded_file = st.file_uploader('テキストをアップロードしてください', type='txt')
    
    if uploaded_file:
        # メタデータ入力フォーム
        with st.expander("メタデータ入力", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                municipality = st.text_input("市区町村名", "")
                major_category = st.selectbox(
                    "大カテゴリ",
                    MAJOR_CATEGORIES
                )
                medium_category = st.selectbox(
                    "中カテゴリ",
                    MEDIUM_CATEGORIES.get(major_category, [])
                )
            
            with col2:
                source = st.text_input("ソース元", "")
                date_time = st.date_input("登録日時", value=datetime.date.today())
                publication_date = st.date_input("データ公開日", value=None)
                latitude = st.text_input("緯度", "")
                longitude = st.text_input("経度", "")
        
        # 登録ボタン
        if st.button("登録する"):
            with st.spinner('登録中...'):
                # メタデータの作成
                metadata = {
                    "municipality": municipality,
                    "major_category": major_category,
                    "medium_category": medium_category,
                    "source": source,
                    "registration_date": str(date_time) if date_time else "",
                    "publication_date": str(publication_date) if publication_date else "",
                    "latitude": latitude,
                    "longitude": longitude,
                }
                
                # ドキュメント登録関数を呼び出し
                register_document(uploaded_file, additional_metadata=metadata)

    st.markdown("---")

    # 2.登録状況確認
    st.subheader("ChromaDB 登録状況確認")
    
    # 検索フィルター
    with st.expander("検索フィルター", expanded=False):
        filter_municipality = st.text_input("市区町村名で絞り込み", "")
        filter_category = st.text_input("カテゴリで絞り込み", "")
    
    # 表示ボタン
    if st.button("登録済みドキュメントを表示"):
        with st.spinner('取得中...'):
            try:
                # グローバルのVectorStoreインスタンスを使用
                dict_data = vector_store.get_documents(ids=None)
                
                if dict_data and len(dict_data.get('ids', [])) > 0:
                    # フィルタリング
                    filtered_indices = range(len(dict_data['ids']))
                    
                    if filter_municipality or filter_category:
                        filtered_indices = []
                        for i, metadata in enumerate(dict_data['metadatas']):
                            municipality_match = True
                            category_match = True
                            
                            if filter_municipality and metadata.get('municipality'):
                                municipality_match = filter_municipality.lower() in metadata['municipality'].lower()
                            
                            if filter_category:
                                major_match = metadata.get('major_category') and filter_category.lower() in metadata['major_category'].lower()
                                medium_match = metadata.get('medium_category') and filter_category.lower() in metadata['medium_category'].lower()
                                category_match = major_match or medium_match
                                
                            if municipality_match and category_match:
                                filtered_indices.append(i)
                    
                    # フィルタリングされたデータでDataFrameを作成
                    filtered_ids = [dict_data['ids'][i] for i in filtered_indices]
                    filtered_docs = [dict_data['documents'][i] for i in filtered_indices]
                    filtered_metas = [dict_data['metadatas'][i] for i in filtered_indices]
                    
                    tmp_df = pd.DataFrame({
                        "IDs": filtered_ids,
                        "Documents": filtered_docs,
                        "市区町村": [m.get('municipality', '') for m in filtered_metas],
                        "大カテゴリ": [m.get('major_category', '') for m in filtered_metas],
                        "中カテゴリ": [m.get('medium_category', '') for m in filtered_metas],
                        "ソース元": [m.get('source', '') for m in filtered_metas],
                        "登録日時": [m.get('registration_date', '') for m in filtered_metas],
                        "データ公開日": [m.get('publication_date', '') for m in filtered_metas],
                        "緯度経度": [f"{m.get('latitude', '')}, {m.get('longitude', '')}" for m in filtered_metas]
                    })
                    
                    st.dataframe(tmp_df)
                    st.success(f"合計 {len(filtered_ids)} 件のドキュメントが表示されています（全 {len(dict_data['ids'])} 件中）")
                else:
                    st.info("データベースに登録されたデータはありません。")
            except Exception as e:
                st.error(f"データの取得中にエラーが発生しました: {e}")
                st.error("エラーの詳細:")
                st.exception(e)

    st.markdown("---")

    # 3.全データ削除
    st.subheader("ChromaDB 登録データ全削除")
    if st.button("全データを削除する"):
        with st.spinner('削除中...'):
            try:
                # グローバルのVectorStoreインスタンスを使用
                current_ids = vector_store.get_documents(ids=None).get('ids', [])
                if current_ids:
                    vector_store.delete_documents(ids=current_ids)
                    st.success(f"データベースから {len(current_ids)} 件のドキュメントが削除されました")
                else:
                    st.info("削除するデータがありません。")
            except Exception as e:
                st.error(f"データの削除中にエラーが発生しました: {e}")
                st.error("エラーの詳細:")
                st.exception(e)

# RAGを使ったLLM回答生成
def generate_response(query_text, filter_conditions=None):
    """
    質問に対する回答を生成する関数。
    filter_conditions: メタデータによるフィルタリング条件
    """
    if not vector_store_available:
        return "申し訳ありません。現在、ベクトルデータベースに接続できないため、質問に回答できません。"
    
    if query_text:
        try:
            # グローバルのVectorStoreインスタンスを使用
            global vector_store

            # カスタムプロンプトテンプレートを作成
            try:
                # まずhub.pullを試す（langchainhubがインストールされている場合）
                prompt = hub.pull("rlm/rag-prompt")
                print("Successfully pulled prompt from langchain hub")
            except (ImportError, Exception) as e:
                # 失敗した場合は自前のプロンプトを使用
                print(f"Using custom prompt template due to: {e}")
                prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # 検索結果を取得（フィルタリング条件があれば適用）
            search_results = vector_store.search(query_text, n_results=5, filter_conditions=filter_conditions)
            
            # 検索結果がない場合
            if not search_results or not search_results.get('documents', [[]])[0]:
                return "申し訳ありません。指定された条件に一致するドキュメントが見つかりませんでした。検索条件を変更してお試しください。"
            
            # 検索結果をドキュメント形式に変換
            from langchain_core.documents import Document
            docs = []
            for i, doc_text in enumerate(search_results['documents'][0]):
                # メタデータの取得（利用可能な場合）
                metadata = {}
                if search_results.get('metadatas') and len(search_results['metadatas']) > 0:
                    metadata = search_results['metadatas'][0][i] if i < len(search_results['metadatas'][0]) else {}
                
                # ドキュメントの作成
                doc = Document(
                    page_content=doc_text,
                    metadata=metadata
                )
                docs.append(doc)

            # 使用するメタデータの情報を表示
            st.markdown("#### 検索結果")
            meta_info = []
            for i, doc in enumerate(docs):
                meta_str = ""
                if doc.metadata.get('municipality'):
                    meta_str += f"【市区町村】{doc.metadata['municipality']} "
                if doc.metadata.get('major_category'):
                    meta_str += f"【大カテゴリ】{doc.metadata['major_category']} "
                if doc.metadata.get('medium_category'):
                    meta_str += f"【中カテゴリ】{doc.metadata['medium_category']} "
                if doc.metadata.get('source'):
                    meta_str += f"【ソース元】{doc.metadata['source']}"
                
                meta_info.append(f"{i+1}. {meta_str}")
            
            st.markdown("\n".join(meta_info))

            qa_chain = (
                {
                    "context": lambda x: format_docs(docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            return qa_chain.invoke(query_text)
        except Exception as e:
            st.error(f"質問の処理中にエラーが発生しました: {e}")
            st.error("エラーの詳細:")
            st.exception(e)
            return None

def ask_question():
    """
    質問するページの関数。
    """
    st.header("ドキュメントに質問する")

    if not vector_store_available:
        st.error("ChromaDBの接続でエラーが発生しました。現在、ベクトルデータベースは使用できません。")
        st.warning("これはSQLiteのバージョンの非互換性によるものです。Streamlit Cloudでの実行には制限があります。")
        st.info("ローカル環境での実行をお試しください。")
        return

    # フィルタリング条件の設定
    with st.expander("検索範囲の絞り込み", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            filter_municipality = st.text_input("市区町村名", "")
            filter_major_category = st.selectbox(
                "大カテゴリ",
                [""] + MAJOR_CATEGORIES
            )
        
        with col2:
            filter_medium_category = st.selectbox(
                "中カテゴリ",
                [""] + (MEDIUM_CATEGORIES.get(filter_major_category, []) if filter_major_category else [])
            )
            filter_source = st.text_input("ソース元", "")

    # Query text
    query_text = st.text_input('質問を入力:', 
                               placeholder='簡単な概要を記入してください')

    # 質問送信ボタン
    if st.button('Submit') and query_text:
        with st.spinner('回答を生成中...'):
            # フィルタリング条件の作成
            filter_conditions = {}
            if filter_municipality:
                filter_conditions["municipality"] = filter_municipality
            if filter_major_category:
                filter_conditions["major_category"] = filter_major_category
            if filter_medium_category:
                filter_conditions["medium_category"] = filter_medium_category
            if filter_source:
                filter_conditions["source"] = filter_source
                
            response = generate_response(query_text, filter_conditions)
            if response:
                st.success("回答:")
                st.info(response)
            else:
                st.error("回答の生成に失敗しました。")

def fallback_mode():
    """
    ChromaDBが使用できない場合のフォールバックモード
    """
    st.header("ChromaDBが使用できません")
    st.error("SQLiteのバージョンの問題により、ChromaDBを使用できません。")
    st.info("このアプリは、SQLite 3.35.0以上が必要です。Streamlit Cloudでは現在、SQLite 3.34.1が使用されています。")
    
    st.markdown("""
    ## 解決策
    
    1. **ローカル環境での実行**: 
       - このアプリをローカル環境でクローンして実行してください
       - 最新のSQLiteがインストールされていることを確認してください
    
    2. **代替のベクトルデータベース**:
       - ChromaDBの代わりに、他のベクトルデータベース（FAISS、Milvusなど）を使用することも検討できます
    
    3. **インメモリモードでの使用**:
       - 現在、DuckDB+Parquetバックエンドでの実行を試みていますが、これも失敗しています
       - 詳細については、ログを確認してください
    """)
    
    # 技術的な詳細
    with st.expander("技術的な詳細"):
        st.code("""
# エラーの原因
ChromaDBは内部でSQLite 3.35.0以上を必要としていますが、
Streamlit Cloudでは現在、SQLite 3.34.1が使用されています。

# 試みた解決策
1. pysqlite3-binaryのインストール
2. SQLiteのソースからのビルド
3. DuckDB+Parquetバックエンドの使用
4. モンキーパッチの適用

いずれも環境制限により成功していません。
        """)

def main():
    """
    アプリケーションのメイン関数。
    """
    # タイトルを表示
    st.title('🦜🔗 Ask the Doc App')

    if not vector_store_available:
        fallback_mode()
        return

    # サイドバーでページ選択
    st.sidebar.title("メニュー")
    page = st.sidebar.radio("ページを選択してください", ["ChromaDB 管理", "質問する",])

    # 各ページへ移動
    if page == "質問する":
        ask_question()
    elif page == "ChromaDB 管理":
        manage_chromadb()

if __name__ == "__main__":
    main()
