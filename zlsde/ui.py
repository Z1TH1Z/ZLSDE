"""Web UI for ZLSDE Pipeline using Gradio."""

import gradio as gr
import pandas as pd
import json
import os
import tempfile
from pathlib import Path
import logging

from zlsde.orchestrator import PipelineOrchestrator
from zlsde.models.data_models import PipelineConfig, DataSource
from zlsde.utils.seed_control import set_random_seed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(
    input_file,
    modality,
    embedding_model,
    clustering_method,
    min_cluster_size,
    use_llm,
    llm_model,
    max_iterations,
    output_format,
    device,
    random_seed
):
    """Run ZLSDE pipeline with user-provided parameters."""
    
    try:
        # Validate input file
        if input_file is None:
            return "❌ Error: Please upload a data file", None, None, None
        
        # Create temporary output directory
        output_dir = tempfile.mkdtemp(prefix="zlsde_output_")
        
        # Determine file type
        file_path = input_file.name
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            source_type = 'csv'
        elif file_ext == '.json':
            source_type = 'json'
        elif file_ext == '.txt':
            source_type = 'text'
        else:
            return f"❌ Error: Unsupported file type: {file_ext}", None, None, None
        
        # Create configuration
        config = PipelineConfig(
            data_sources=[DataSource(type=source_type, path=file_path)],
            modality=modality,
            embedding_model=embedding_model,
            clustering_method=clustering_method,
            min_cluster_size=int(min_cluster_size),
            llm_model=llm_model,
            use_llm=use_llm,
            max_iterations=int(max_iterations),
            output_format=output_format,
            output_path=output_dir,
            device=device,
            random_seed=int(random_seed),
            log_level="INFO"
        )
        
        # Create and run pipeline
        orchestrator = PipelineOrchestrator(config)
        
        result = orchestrator.run()
        
        if result.status != "completed":
            return f"❌ Pipeline failed: {result.error_message}", None, None, None
        
        # Load results
        dataset_path = result.dataset_path
        metadata_path = os.path.join(output_dir, "metadata.json")
        
        # Read dataset
        if output_format == "csv":
            df = pd.read_csv(dataset_path)
        elif output_format == "json":
            df = pd.read_json(dataset_path)
        elif output_format == "parquet":
            df = pd.read_parquet(dataset_path)
        
        # Read metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create summary
        summary = f"""
✅ **Pipeline Completed Successfully!**

📊 **Results Summary:**
- Total Samples: {result.n_samples}
- Labeled Samples: {result.n_labeled}
- Number of Clusters: {result.final_metrics.n_clusters}
- Silhouette Score: {result.final_metrics.silhouette_score:.3f}
- Quality Mean: {result.final_metrics.quality_mean:.3f}
- Execution Time: {result.execution_time_seconds:.2f} seconds

📈 **Iteration History:**
"""
        for i, metrics in enumerate(result.iteration_history):
            summary += f"\n- Iteration {i+1}: {metrics.n_clusters} clusters, silhouette={metrics.silhouette_score:.3f}, flip_rate={metrics.label_flip_rate:.1%}"
        
        summary += f"""

📁 **Output Files:**
- Dataset: {dataset_path}
- Metadata: {metadata_path}
"""
        
        # Create metadata display
        metadata_display = json.dumps(metadata, indent=2)
        
        return summary, df, metadata_display, dataset_path
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return f"❌ Error: {str(e)}", None, None, None


def create_ui():
    """Create Gradio UI for ZLSDE Pipeline."""
    
    with gr.Blocks(title="ZLSDE - Zero-Label Dataset Engine", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🚀 ZLSDE - Zero-Label Self-Discovering Dataset Engine
        
        Transform raw unlabeled data into structured, labeled datasets **without human annotation**!
        
        Upload your data, configure the pipeline, and let AI do the labeling for you.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📤 Input Configuration")
                
                # File upload
                input_file = gr.File(
                    label="Upload Data File",
                    file_types=[".csv", ".json", ".txt"],
                    type="filepath"
                )
                
                gr.Markdown("**Supported formats:** CSV (with 'content' column), JSON, TXT")
                
                # Basic settings
                with gr.Accordion("⚙️ Basic Settings", open=True):
                    modality = gr.Dropdown(
                        choices=["text", "image", "multimodal"],
                        value="text",
                        label="Data Modality",
                        info="Type of data you're processing"
                    )
                    
                    output_format = gr.Dropdown(
                        choices=["csv", "json", "parquet"],
                        value="csv",
                        label="Output Format",
                        info="Format for the labeled dataset"
                    )
                    
                    device = gr.Dropdown(
                        choices=["cpu", "cuda"],
                        value="cpu",
                        label="Device",
                        info="Use GPU (cuda) for faster processing if available"
                    )
                
                # Embedding settings
                with gr.Accordion("🧠 Embedding Settings", open=False):
                    embedding_model = gr.Dropdown(
                        choices=[
                            "sentence-transformers/all-MiniLM-L6-v2",
                            "sentence-transformers/all-mpnet-base-v2",
                            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                        ],
                        value="sentence-transformers/all-MiniLM-L6-v2",
                        label="Embedding Model",
                        info="Model for generating embeddings (smaller = faster)"
                    )
                
                # Clustering settings
                with gr.Accordion("🔍 Clustering Settings", open=False):
                    clustering_method = gr.Dropdown(
                        choices=["auto", "kmeans", "spectral"],
                        value="auto",
                        label="Clustering Method",
                        info="Auto tries multiple methods and picks the best"
                    )
                    
                    min_cluster_size = gr.Slider(
                        minimum=2,
                        maximum=50,
                        value=5,
                        step=1,
                        label="Minimum Cluster Size",
                        info="Minimum samples per cluster"
                    )
                
                # Label generation settings
                with gr.Accordion("🏷️ Label Generation Settings", open=False):
                    use_llm = gr.Checkbox(
                        value=True,
                        label="Use LLM for Label Generation",
                        info="Generate semantic labels using AI (slower but better)"
                    )
                    
                    llm_model = gr.Dropdown(
                        choices=[
                            "google/flan-t5-base",
                            "google/flan-t5-small",
                            "google/flan-t5-large"
                        ],
                        value="google/flan-t5-base",
                        label="LLM Model",
                        info="Language model for generating labels"
                    )
                
                # Training settings
                with gr.Accordion("🔄 Training Settings", open=False):
                    max_iterations = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Max Iterations",
                        info="Number of self-training refinement iterations"
                    )
                    
                    random_seed = gr.Number(
                        value=42,
                        label="Random Seed",
                        info="For reproducible results"
                    )
                
                # Run button
                run_btn = gr.Button("🚀 Run Pipeline", variant="primary", size="lg")
                
                gr.Markdown("""
                ---
                **💡 Tips:**
                - Start with default settings for best results
                - Larger datasets may take several minutes
                - Use GPU (cuda) for faster processing if available
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("## 📊 Results")
                
                # Results display
                summary_output = gr.Markdown(label="Summary")
                
                with gr.Tabs():
                    with gr.Tab("📋 Labeled Dataset"):
                        dataset_output = gr.Dataframe(
                            label="Labeled Data",
                            interactive=False,
                            wrap=True
                        )
                    
                    with gr.Tab("📈 Metadata"):
                        metadata_output = gr.Code(
                            label="Pipeline Metadata",
                            language="json"
                        )
                    
                    with gr.Tab("💾 Download"):
                        download_output = gr.File(label="Download Dataset")
                        gr.Markdown("""
                        **Download your labeled dataset:**
                        - Click the download button above to save the file
                        - The dataset includes labels, confidence scores, and quality metrics
                        """)
        
        # Connect button to function
        run_btn.click(
            fn=run_pipeline,
            inputs=[
                input_file,
                modality,
                embedding_model,
                clustering_method,
                min_cluster_size,
                use_llm,
                llm_model,
                max_iterations,
                output_format,
                device,
                random_seed
            ],
            outputs=[
                summary_output,
                dataset_output,
                metadata_output,
                download_output
            ]
        )
        
        # Examples
        gr.Markdown("""
        ---
        ## 📚 Example Datasets
        
        Try these example files to test the pipeline:
        - `examples/data/sample_text.csv` - 20 technology-related text samples
        
        ## 🎯 How It Works
        
        1. **Upload** your unlabeled data (CSV, JSON, or TXT)
        2. **Configure** the pipeline settings (or use defaults)
        3. **Run** the pipeline and wait for results
        4. **Download** your labeled dataset with quality scores
        
        The pipeline automatically:
        - Generates embeddings from your data
        - Discovers natural clusters
        - Generates semantic labels using AI
        - Applies quality control
        - Refines labels through self-training
        """)
    
    return app


def main():
    """Launch the Gradio UI."""
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
