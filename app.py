"""
DETR Streamlit Visualization Application
Interactive web application for DETR model visualization and analysis.
"""

import streamlit as st
import time
import os
import tempfile
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import sys
sys.path.append('src')
from detr_model import DETRModelManager, DETRVisualizer
from sample_images import SampleImageManager

# Page configuration
st.set_page_config(
    page_title="DETR Visualization",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
    }
    .detection-box {
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .attention-map {
        border: 2px solid #764ba2;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


class DETRApp:
    """DETR Application for visualization and analysis"""
    
    def __init__(self):
        """Initialize the application"""
        # Initialize session state
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'current_image' not in st.session_state:
            st.session_state.current_image = None
        if 'detection_results' not in st.session_state:
            st.session_state.detection_results = None
        if 'attention_maps' not in st.session_state:
            st.session_state.attention_maps = None
        if 'query_analysis' not in st.session_state:
            st.session_state.query_analysis = None
        if 'model_manager' not in st.session_state:
            st.session_state.model_manager = DETRModelManager()
        
        # Use session state for model manager
        self.model_manager = st.session_state.model_manager
        self.visualizer = DETRVisualizer()
        self.sample_manager = SampleImageManager()
        
        # Note: Model will be loaded manually via sidebar button
        # Auto-loading is disabled to prevent errors during initialization
    
    def run(self):
        """Run the application"""
        self._display_header()
        self._display_sidebar()
        self._display_main_content()
    
    def _display_header(self):
        """Display the application header"""
        st.markdown("""
        <div class="main-header">
            <h1>🎯 DETR Visualization</h1>
            <p>DEtection TRansformer - End-to-End Object Detection with Transformers</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_sidebar(self):
        """Display the sidebar with model configuration"""
        st.sidebar.title("🔧 設定")
        
        # Model configuration
        st.sidebar.subheader("🤖 モデル設定")
        
        # Model selection
        model_options = {
            "facebook/detr-resnet-50": "DETR ResNet-50 (Object Detection)",
            "facebook/detr-resnet-101": "DETR ResNet-101 (Object Detection)",
            "facebook/detr-resnet-50-panoptic": "DETR ResNet-50 (Panoptic Segmentation)",
        }
        
        selected_model = st.sidebar.selectbox(
            "DETRモデルを選択",
            list(model_options.keys()),
            index=0,
            format_func=lambda x: model_options[x],
            help="使用するDETRモデルを選択してください"
        )
        
        # Task selection
        task_options = ["object-detection", "segmentation"]
        selected_task = st.sidebar.selectbox(
            "タスクを選択",
            task_options,
            index=0,
            help="実行するタスクを選択してください"
        )
        
        # Confidence threshold
        confidence_threshold = st.sidebar.slider(
            "信頼度閾値",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="物体検知の信頼度閾値を設定してください"
        )
        
        # Load model button
        if not st.session_state.model_loaded:
            if st.sidebar.button("モデルを読み込み", type="primary"):
                with st.spinner("DETRモデルを読み込み中..."):
                    try:
                        # Update model manager configuration
                        self.model_manager.model_name = selected_model
                        self.model_manager.task = selected_task
                        
                        # Load model
                        success = self.model_manager.load_model()
                        if success:
                            st.session_state.model_loaded = True
                            st.sidebar.success("✅ モデル読み込み完了")
                        else:
                            st.sidebar.error("❌ モデル読み込みに失敗しました")
                    except Exception as e:
                        st.sidebar.error(f"❌ モデル読み込みエラー: {str(e)}")
                        st.sidebar.error("詳細はターミナルログを確認してください")
        else:
            st.sidebar.success("✅ モデル読み込み済み")
            if st.sidebar.button("モデルを再読み込み"):
                with st.spinner("DETRモデルを再読み込み中..."):
                    try:
                        # Update model manager configuration
                        self.model_manager.model_name = selected_model
                        self.model_manager.task = selected_task
                        
                        # Load model
                        success = self.model_manager.load_model()
                        if success:
                            st.sidebar.success("✅ モデル再読み込み完了")
                        else:
                            st.sidebar.error("❌ モデル再読み込みに失敗しました")
                    except Exception as e:
                        st.sidebar.error(f"❌ モデル再読み込みエラー: {str(e)}")
                        st.sidebar.error("詳細はターミナルログを確認してください")
        
        # Model info
        if st.session_state.model_loaded:
            try:
                model_info = self.model_manager.get_model_info()
                if "error" not in model_info:
                    st.sidebar.subheader("📊 モデル情報")
                    st.sidebar.write(f"**モデル:** {model_info.get('model_name', 'Unknown')}")
                    st.sidebar.write(f"**タスク:** {model_info.get('task', 'Unknown')}")
                    st.sidebar.write(f"**デバイス:** {model_info.get('device', 'Unknown')}")
                    st.sidebar.write(f"**パラメータ数:** {model_info.get('total_parameters', 0):,}")
                else:
                    st.sidebar.error(f"❌ モデル情報の取得に失敗: {model_info['error']}")
            except Exception as e:
                st.sidebar.error(f"❌ モデル情報の取得に失敗: {str(e)}")
        else:
            st.sidebar.warning("⚠️ モデルが読み込まれていません")
        
        # Image selection
        st.sidebar.subheader("🖼️ 画像選択")
        
        # Sample images
        sample_images = self.sample_manager.get_sample_images()
        
        # Download sample images if not available
        if not any(img["exists"] for img in sample_images.values()):
            if st.sidebar.button("サンプル画像をダウンロード"):
                with st.spinner("サンプル画像をダウンロード中..."):
                    downloaded = self.sample_manager.download_sample_images()
                    if downloaded:
                        st.sidebar.success(f"✅ {len(downloaded)}枚の画像をダウンロードしました")
                        st.rerun()
        
        # Sample image selection
        available_samples = {name: info for name, info in sample_images.items() if info["exists"]}
        
        if available_samples:
            selected_sample = st.sidebar.selectbox(
                "サンプル画像を選択",
                list(available_samples.keys()),
                format_func=lambda x: f"{x} - {available_samples[x]['description']}"
            )
            
            if st.sidebar.button("サンプル画像を読み込み"):
                image = self.sample_manager.load_sample_image(selected_sample)
                if image:
                    st.session_state.current_image = image
                    st.sidebar.success(f"✅ {selected_sample}画像を読み込みました")
        
        # Upload image
        uploaded_file = st.sidebar.file_uploader(
            "画像ファイルをアップロード",
            type=['png', 'jpg', 'jpeg'],
            help="独自の画像をアップロードできます"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.session_state.current_image = image
            st.sidebar.success("✅ 画像をアップロードしました")
        
        # Synthetic image
        if st.sidebar.button("合成画像を生成"):
            image = self.sample_manager.create_synthetic_image()
            st.session_state.current_image = image
            st.sidebar.success("✅ 合成画像を生成しました")
    
    def _display_main_content(self):
        """Display the main content area"""
        if not st.session_state.model_loaded:
            st.warning("⚠️ まずサイドバーでDETRモデルを読み込んでください。")
            return
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 物体検知", 
            "🔍 アテンション可視化", 
            "📊 クエリ分析", 
            "🏗️ アーキテクチャ",
            "📈 内部状態",
            "⚙️ デバッグ情報"
        ])
        
        with tab1:
            self._display_object_detection_tab()
        
        with tab2:
            self._display_attention_visualization_tab()
        
        with tab3:
            self._display_query_analysis_tab()
        
        with tab4:
            self._display_architecture_tab()
        
        with tab5:
            self._display_internal_state_tab()
        
        with tab6:
            self._display_debug_tab()
    
    def _display_object_detection_tab(self):
        """Display object detection tab"""
        st.header("🎯 物体検知")
        
        if st.session_state.current_image is None:
            st.warning("⚠️ 画像が選択されていません。サイドバーで画像を選択してください。")
            return
        
        # Display current image
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📸 入力画像")
            st.image(st.session_state.current_image, caption="入力画像", use_container_width=True)
            
            # Image info
            image_info = self.sample_manager.get_image_info(st.session_state.current_image)
            if "error" not in image_info:
                st.write(f"**サイズ:** {image_info['size']}")
                st.write(f"**アスペクト比:** {image_info['aspect_ratio']:.2f}")
        
        with col2:
            st.subheader("⚙️ 検知設定")
            
            # Confidence threshold
            confidence_threshold = st.slider(
                "信頼度閾値",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1
            )
            
            # Detection button
            if st.button("物体検知を実行", type="primary"):
                with st.spinner("物体検知を実行中..."):
                    try:
                        results = self.model_manager.detect_objects(
                            st.session_state.current_image, 
                            confidence_threshold
                        )
                        st.session_state.detection_results = results
                        st.success(f"✅ {results['num_detections']}個の物体を検知しました")
                    except Exception as e:
                        st.error(f"❌ 物体検知に失敗しました: {str(e)}")
        
        # Display detection results
        if st.session_state.detection_results:
            st.subheader("🔍 検知結果")
            
            results = st.session_state.detection_results
            detections = results["detections"]
            
            # Detection statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("検知数", results["num_detections"])
            with col2:
                if detections:
                    avg_confidence = np.mean([d["score"] for d in detections])
                    st.metric("平均信頼度", f"{avg_confidence:.3f}")
            with col3:
                if detections:
                    max_confidence = max([d["score"] for d in detections])
                    st.metric("最大信頼度", f"{max_confidence:.3f}")
            with col4:
                if detections:
                    unique_classes = len(set([d["label_name"] for d in detections]))
                    st.metric("クラス数", unique_classes)
            
            # Detection visualization
            fig = self.visualizer.visualize_detections(
                st.session_state.current_image, 
                detections,
                "DETR物体検知結果"
            )
            st.pyplot(fig)
            
            # Detection details
            st.subheader("📋 検知詳細")
            
            if detections:
                # Create DataFrame for detections
                detection_data = []
                for i, detection in enumerate(detections):
                    detection_data.append({
                        "ID": i + 1,
                        "クラス": detection["label_name"],
                        "信頼度": f"{detection['score']:.3f}",
                        "バウンディングボックス": f"[{detection['box'][0]:.1f}, {detection['box'][1]:.1f}, {detection['box'][2]:.1f}, {detection['box'][3]:.1f}]",
                        "中心座標": f"[{detection['center'][0]:.1f}, {detection['center'][1]:.1f}]",
                        "幅": f"{detection['width']:.1f}",
                        "高さ": f"{detection['height']:.1f}",
                    })
                
                df = pd.DataFrame(detection_data)
                st.dataframe(df, use_container_width=True)
                
                # Class distribution
                st.subheader("📊 クラス分布")
                class_counts = {}
                for detection in detections:
                    class_name = detection["label_name"]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                if class_counts:
                    fig = px.bar(
                        x=list(class_counts.keys()),
                        y=list(class_counts.values()),
                        title="検知されたクラスの分布",
                        labels={"x": "クラス", "y": "検知数"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def _display_attention_visualization_tab(self):
        """Display attention visualization tab"""
        st.header("🔍 アテンション可視化")
        
        if st.session_state.current_image is None:
            st.warning("⚠️ 画像が選択されていません。サイドバーで画像を選択してください。")
            return
        
        # Get attention maps
        if st.button("アテンションマップを取得", type="primary"):
            with st.spinner("アテンションマップを取得中..."):
                try:
                    attention_maps = self.model_manager.get_attention_maps(st.session_state.current_image)
                    st.session_state.attention_maps = attention_maps
                    st.success("✅ アテンションマップを取得しました")
                except Exception as e:
                    st.error(f"❌ アテンションマップの取得に失敗しました: {str(e)}")
        
        if st.session_state.attention_maps:
            attention_maps = st.session_state.attention_maps
            
            # Attention type selection
            attention_types = []
            if attention_maps["encoder_attentions"]:
                attention_types.append("encoder")
            if attention_maps["decoder_attentions"]:
                attention_types.append("decoder")
            if attention_maps["cross_attentions"]:
                attention_types.append("cross")
            
            if attention_types:
                selected_type = st.selectbox(
                    "アテンションタイプを選択",
                    attention_types,
                    help="可視化するアテンションタイプを選択してください"
                )
                
                # Layer selection
                attentions = attention_maps[f"{selected_type}_attentions"]
                max_layer = len(attentions) - 1
                
                selected_layer = st.slider(
                    "レイヤーを選択",
                    min_value=0,
                    max_value=max_layer,
                    value=0,
                    help="可視化するレイヤーを選択してください"
                )
                
                # Visualize attention map
                fig = self.visualizer.visualize_attention_maps(
                    attention_maps, 
                    selected_type, 
                    selected_layer
                )
                
                if fig:
                    st.pyplot(fig)
                    
                    # Attention statistics
                    attention_matrix = attentions[selected_layer]["attention"]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("平均アテンション", f"{np.mean(attention_matrix):.4f}")
                    with col2:
                        st.metric("最大アテンション", f"{np.max(attention_matrix):.4f}")
                    with col3:
                        st.metric("最小アテンション", f"{np.min(attention_matrix):.4f}")
                    with col4:
                        st.metric("標準偏差", f"{np.std(attention_matrix):.4f}")
                    
                    # Attention distribution
                    st.subheader("📊 アテンション分布")
                    fig = px.histogram(
                        x=attention_matrix.flatten(),
                        title=f"{selected_type.capitalize()} Attention Distribution - Layer {selected_layer}",
                        labels={"x": "Attention Weight", "y": "Frequency"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def _display_query_analysis_tab(self):
        """Display query analysis tab"""
        st.header("📊 クエリ分析")
        
        if st.session_state.current_image is None:
            st.warning("⚠️ 画像が選択されていません。サイドバーで画像を選択してください。")
            return
        
        # Get query analysis
        if st.button("クエリ分析を実行", type="primary"):
            with st.spinner("クエリ分析を実行中..."):
                try:
                    query_analysis = self.model_manager.get_object_queries_analysis(st.session_state.current_image)
                    st.session_state.query_analysis = query_analysis
                    st.success("✅ クエリ分析を完了しました")
                except Exception as e:
                    st.error(f"❌ クエリ分析に失敗しました: {str(e)}")
        
        if st.session_state.query_analysis:
            query_analysis = st.session_state.query_analysis
            
            # Query analysis visualization
            fig = self.visualizer.visualize_query_analysis(query_analysis)
            if fig:
                st.pyplot(fig)
            
            # Query details
            st.subheader("📋 クエリ詳細")
            
            predictions = query_analysis["query_predictions"]
            
            # Top queries
            top_queries = sorted(predictions, key=lambda x: x["confidence"], reverse=True)[:10]
            
            query_data = []
            for query in top_queries:
                query_data.append({
                    "クエリID": query["query_idx"],
                    "予測クラス": query["predicted_class"],
                    "信頼度": f"{query['confidence']:.3f}",
                    "バウンディングボックス": f"[{query['box'][0]:.3f}, {query['box'][1]:.3f}, {query['box'][2]:.3f}, {query['box'][3]:.3f}]",
                })
            
            df = pd.DataFrame(query_data)
            st.dataframe(df, use_container_width=True)
    
    def _display_architecture_tab(self):
        """Display architecture tab"""
        st.header("🏗️ DETRアーキテクチャ")
        
        # Architecture diagram
        st.subheader("📐 アーキテクチャ図")
        fig = self.visualizer.create_detr_architecture_diagram()
        st.pyplot(fig)
        
        # Architecture explanation
        st.subheader("📖 アーキテクチャ説明")
        
        st.markdown("""
        ### DETRの主要コンポーネント
        
        1. **CNNバックボーン (ResNet-50/101)**
           - 入力画像から特徴マップを抽出
           - 空間解像度を1/32に縮小
           - チャンネル数を256に投影
        
        2. **Transformerエンコーダー**
           - 6層の自己アテンション層
           - 画像特徴のグローバルな関係性を学習
           - 位置エンコーディングを追加
        
        3. **Transformerデコーダー**
           - 6層のデコーダー層
           - 100個のオブジェクトクエリを使用
           - エンコーダー出力とのクロスアテンション
        
        4. **予測ヘッド**
           - 分類ヘッド: 91クラス + 背景
           - バウンディングボックスヘッド: 4次元 (中心x, y, 幅, 高さ)
        
        ### 重要な特徴
        
        - **End-to-End学習**: NMSやアンカーボックスが不要
        - **セット予測**: 固定数のクエリで物体を予測
        - **二部マッチング損失**: 予測と正解の最適な対応付け
        - **Transformer**: グローバルなコンテキストを活用
        """)
        
        # Model configuration
        if st.session_state.model_loaded:
            st.subheader("⚙️ モデル設定")
            try:
                model_info = self.model_manager.get_model_info()
                if "error" not in model_info and "model_config" in model_info:
                    config = model_info["model_config"]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**基本設定:**")
                        st.write(f"• クエリ数: {config.get('num_queries', 'Unknown')}")
                        st.write(f"• 隠れ層サイズ: {config.get('hidden_size', 'Unknown')}")
                        st.write(f"• クラス数: {config.get('num_labels', 'Unknown')}")
                    
                    with col2:
                        st.write("**Transformer設定:**")
                        st.write(f"• エンコーダー層数: {config.get('num_encoder_layers', 'Unknown')}")
                        st.write(f"• デコーダー層数: {config.get('num_decoder_layers', 'Unknown')}")
                        st.write(f"• アテンションヘッド数: {config.get('num_attention_heads', 'Unknown')}")
                else:
                    st.error(f"❌ モデル設定の取得に失敗: {model_info.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ モデル設定の取得に失敗: {str(e)}")
    
    def _display_internal_state_tab(self):
        """Display internal state tab"""
        st.header("📈 内部状態")
        
        if st.session_state.current_image is None:
            st.warning("⚠️ 画像が選択されていません。サイドバーで画像を選択してください。")
            return
        
        if not st.session_state.attention_maps:
            st.warning("⚠️ アテンションマップが取得されていません。アテンション可視化タブで取得してください。")
            return
        
        attention_maps = st.session_state.attention_maps
        
        # Hidden states analysis
        st.subheader("🔍 隠れ状態分析")
        
        if attention_maps["encoder_hidden_states"]:
            st.write("**エンコーダー隠れ状態:**")
            
            # Analyze encoder hidden states
            encoder_states = attention_maps["encoder_hidden_states"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("層数", len(encoder_states))
            
            with col2:
                if encoder_states:
                    mean_activation = np.mean(encoder_states[-1])
                    st.metric("最終層平均活性化", f"{mean_activation:.4f}")
            
            with col3:
                if encoder_states:
                    std_activation = np.std(encoder_states[-1])
                    st.metric("最終層標準偏差", f"{std_activation:.4f}")
            
            # Hidden states visualization
            if len(encoder_states) > 1:
                st.subheader("📊 エンコーダー隠れ状態の変化")
                
                # Calculate statistics for each layer
                layer_stats = []
                for i, hidden_state in enumerate(encoder_states):
                    layer_stats.append({
                        "Layer": i,
                        "Mean": np.mean(hidden_state),
                        "Std": np.std(hidden_state),
                        "Max": np.max(hidden_state),
                        "Min": np.min(hidden_state),
                    })
                
                df = pd.DataFrame(layer_stats)
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Mean Activation", "Standard Deviation", "Max Activation", "Min Activation")
                )
                
                fig.add_trace(go.Scatter(x=df["Layer"], y=df["Mean"], mode="lines+markers"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df["Layer"], y=df["Std"], mode="lines+markers"), row=1, col=2)
                fig.add_trace(go.Scatter(x=df["Layer"], y=df["Max"], mode="lines+markers"), row=2, col=1)
                fig.add_trace(go.Scatter(x=df["Layer"], y=df["Min"], mode="lines+markers"), row=2, col=2)
                
                fig.update_layout(height=600, title_text="Encoder Hidden States Statistics")
                st.plotly_chart(fig, use_container_width=True)
        
        # Decoder hidden states
        if attention_maps["decoder_hidden_states"]:
            st.subheader("🔍 デコーダー隠れ状態")
            
            decoder_states = attention_maps["decoder_hidden_states"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("層数", len(decoder_states))
            
            with col2:
                if decoder_states:
                    mean_activation = np.mean(decoder_states[-1])
                    st.metric("最終層平均活性化", f"{mean_activation:.4f}")
            
            with col3:
                if decoder_states:
                    std_activation = np.std(decoder_states[-1])
                    st.metric("最終層標準偏差", f"{std_activation:.4f}")
    
    def _display_debug_tab(self):
        """Display debug information tab"""
        st.header("⚙️ デバッグ情報")
        
        # Model information
        st.subheader("🤖 モデル情報")
        try:
            model_info = self.model_manager.get_model_info()
            st.json(model_info)
        except Exception as e:
            st.error(f"❌ モデル情報の取得に失敗: {str(e)}")
            st.json({"error": str(e)})
        
        # System information
        st.subheader("💻 システム情報")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**PyTorch情報:**")
            st.write(f"• PyTorch バージョン: {torch.__version__}")
            st.write(f"• CUDA 利用可能: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                st.write(f"• CUDA バージョン: {torch.version.cuda}")
                st.write(f"• GPU 数: {torch.cuda.device_count()}")
                st.write(f"• 現在のGPU: {torch.cuda.get_device_name()}")
        
        with col2:
            st.write("**アプリケーション情報:**")
            st.write(f"• モデル読み込み済み: {st.session_state.model_loaded}")
            st.write(f"• 現在の画像: {st.session_state.current_image is not None}")
            st.write(f"• 検知結果: {st.session_state.detection_results is not None}")
            st.write(f"• アテンションマップ: {st.session_state.attention_maps is not None}")
            st.write(f"• クエリ分析: {st.session_state.query_analysis is not None}")
        
        # Memory usage
        if torch.cuda.is_available():
            st.subheader("💾 GPUメモリ使用量")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                allocated = torch.cuda.memory_allocated() / 1024**3
                st.metric("割り当て済み", f"{allocated:.2f} GB")
            
            with col2:
                cached = torch.cuda.memory_reserved() / 1024**3
                st.metric("キャッシュ済み", f"{cached:.2f} GB")
            
            with col3:
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                st.metric("総メモリ", f"{total:.2f} GB")
        
        # Raw data
        if st.session_state.detection_results:
            st.subheader("🔍 生データ")
            
            with st.expander("検知結果"):
                st.json(st.session_state.detection_results)
        
        if st.session_state.attention_maps:
            with st.expander("アテンションマップ"):
                # Show summary instead of full data
                attention_summary = {
                    "encoder_attentions": len(st.session_state.attention_maps["encoder_attentions"]),
                    "decoder_attentions": len(st.session_state.attention_maps["decoder_attentions"]),
                    "cross_attentions": len(st.session_state.attention_maps["cross_attentions"]),
                    "encoder_hidden_states": len(st.session_state.attention_maps["encoder_hidden_states"]),
                    "decoder_hidden_states": len(st.session_state.attention_maps["decoder_hidden_states"]),
                }
                st.json(attention_summary)


def main():
    """Main function"""
    app = DETRApp()
    app.run()


if __name__ == "__main__":
    main() 