"""
Speaker Diarization Inference với NeMo ClusteringDiarizer
Dựa trên: https://github.com/NVIDIA-NeMo/NeMo/blob/stable/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb
"""
import os
import json
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels, labels_to_pyannote_object


def create_manifest(audio_file: str, manifest_path: str, rttm_file: str = None, num_speakers: int = None):
    """
    Tạo manifest file cho NeMo diarization
    Format: {'audio_filepath': path, 'offset': 0, 'duration': None, 'label': 'infer', 
             'text': '-', 'num_speakers': None, 'rttm_filepath': None, 'uem_filepath': None}
    """
    meta = {
        'audio_filepath': os.path.abspath(audio_file),
        'offset': 0,
        'duration': None,
        'label': 'infer',
        'text': '-',
        'num_speakers': num_speakers,
        'rttm_filepath': os.path.abspath(rttm_file) if rttm_file and os.path.exists(rttm_file) else None,
        'uem_filepath': None
    }
    
    with open(manifest_path, 'w', encoding='utf-8') as fp:
        json.dump(meta, fp)
        fp.write('\n')
    
    print(f"✓ Created manifest: {manifest_path}")
    return manifest_path


def inspect_clustering_results(output_dir: str):
    """
    Kiểm tra và hiển thị kết quả clustering
    """
    print("\n" + "="*80)
    print("CLUSTERING RESULTS ANALYSIS")
    print("="*80)
    
    # 1. Kiểm tra RTTM output
    pred_rttm_dir = os.path.join(output_dir, 'pred_rttms')
    if os.path.exists(pred_rttm_dir):
        rttm_files = [f for f in os.listdir(pred_rttm_dir) if f.endswith('.rttm')]
        print(f"\n[RTTM Files] Found {len(rttm_files)} file(s):")
        for rttm_file in rttm_files:
            rttm_path = os.path.join(pred_rttm_dir, rttm_file)
            print(f"\n  → {rttm_file}")
            print("  " + "-"*60)
            with open(rttm_path, 'r', encoding='utf-8') as f:
                for line in f:
                    print(f"    {line.strip()}")
    
    # 2. Kiểm tra speaker outputs (embeddings, clustering labels)
    speaker_output_dir = os.path.join(output_dir, 'speaker_outputs')
    if os.path.exists(speaker_output_dir):
        print(f"\n[Speaker Outputs] Directory: {speaker_output_dir}")
        files = os.listdir(speaker_output_dir)
        
        # Oracle VAD manifest
        oracle_vad = [f for f in files if 'oracle_vad' in f]
        if oracle_vad:
            print(f"  • Oracle VAD files: {oracle_vad}")
        
        # Subsegments files (chứa thông tin về segments)
        subsegments = [f for f in files if 'subsegments' in f and f.endswith('.json')]
        print(f"  • Subsegment files: {subsegments}")
        
        # Cluster label files (kết quả clustering)
        cluster_labels = [f for f in files if 'cluster.label' in f]
        if cluster_labels:
            print(f"\n[Cluster Labels] Found {len(cluster_labels)} file(s):")
            for label_file in cluster_labels:
                label_path = os.path.join(speaker_output_dir, label_file)
                print(f"\n  → {label_file}")
                print("  " + "-"*60)
                with open(label_path, 'r', encoding='utf-8') as f:
                    labels = f.read().strip().split('\n')
                    print(f"    Total segments: {len(labels)}")
                    print(f"    Unique speakers: {len(set(labels))}")
                    print(f"    Label distribution: {dict((x, labels.count(x)) for x in set(labels))}")
                    print(f"    First 20 labels: {' '.join(labels[:20])}")
        
        # Embeddings files (nếu save_embeddings=True)
        emb_dir = os.path.join(speaker_output_dir, 'embeddings')
        if os.path.exists(emb_dir):
            emb_files = os.listdir(emb_dir)
            print(f"\n[Embeddings] Found {len(emb_files)} file(s) in {emb_dir}")
    
    # 3. Kiểm tra VAD outputs
    vad_output_dir = os.path.join(output_dir, 'vad_outputs')
    if os.path.exists(vad_output_dir):
        print(f"\n[VAD Outputs] Directory: {vad_output_dir}")
        vad_files = os.listdir(vad_output_dir)
        print(f"  • Files: {vad_files}")
    
    print("\n" + "="*80)


def visualize_diarization(rttm_file: str, audio_name: str = "audio"):
    """
    Visualize diarization results from RTTM file
    """
    if not os.path.exists(rttm_file):
        print(f"RTTM file not found: {rttm_file}")
        return
    
    print(f"\n[Visualization] {audio_name}")
    print("-"*80)
    
    # Convert RTTM to labels
    labels = rttm_to_labels(rttm_file)
    print(f"\nTotal segments: {len(labels)}")
    
    # Group by speaker
    speakers = {}
    for label in labels:
        start, end, speaker = label.split()
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append((float(start), float(end)))
    
    print(f"Number of speakers: {len(speakers)}")
    
    for speaker, segments in speakers.items():
        total_duration = sum(end - start for start, end in segments)
        print(f"\n{speaker}:")
        print(f"  - Segments: {len(segments)}")
        print(f"  - Total duration: {total_duration:.2f}s")
        print(f"  - First 5 segments: {segments[:5]}")


def diarize_audio(audio_path: str, config_yaml: str, output_dir: str = None, 
                  rttm_file: str = None, num_speakers: int = None):
    """
    Chạy speaker diarization với NeMo ClusteringDiarizer
    
    Args:
        audio_path: Đường dẫn đến file audio
        config_yaml: Đường dẫn đến config file
        output_dir: Thư mục output (nếu None, sẽ lấy từ config)
        rttm_file: Ground truth RTTM file (optional, dùng để evaluate)
        num_speakers: Số lượng speakers (optional, dùng oracle_num_speakers)
    """
    
    # Load config
    cfg = OmegaConf.load(config_yaml)
    print(f"✓ Loaded config from: {config_yaml}")
    
    # Set output directory
    if output_dir:
        cfg.diarizer.out_dir = output_dir
    os.makedirs(cfg.diarizer.out_dir, exist_ok=True)
    
    # Create manifest file
    manifest_path = os.path.join(cfg.diarizer.out_dir, 'input_manifest.json')
    create_manifest(audio_path, manifest_path, rttm_file, num_speakers)
    
    # Update config with manifest path
    cfg.diarizer.manifest_filepath = manifest_path
    
    # Oracle num speakers nếu được cung cấp
    if num_speakers is not None:
        cfg.diarizer.clustering.parameters.oracle_num_speakers = True
        print(f"✓ Using oracle number of speakers: {num_speakers}")
    
    # Ensure verbose is set
    if 'verbose' not in cfg:
        cfg.verbose = True
    
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(OmegaConf.to_yaml(cfg))
    print("="*80)
    
    # Initialize ClusteringDiarizer
    print("\n[1/4] Initializing ClusteringDiarizer...")
    try:
        diarizer = ClusteringDiarizer(cfg=cfg)
    except Exception as e:
        print(f"Error initializing ClusteringDiarizer: {e}")
        print("\nTrying with OmegaConf.to_container...")
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        diarizer = ClusteringDiarizer(cfg=OmegaConf.create(cfg_dict))
    
    # Run diarization
    print("\n[2/4] Running diarization pipeline...")
    print("  → This includes: VAD → Segmentation → Embedding Extraction → Clustering")
    score = diarizer.diarize()
    
    print(f"\n[3/4] Diarization completed!")
    if score is not None:
        print(f"  → DER (Diarization Error Rate): {score}")
    
    # Inspect clustering results
    print("\n[4/4] Inspecting results...")
    inspect_clustering_results(cfg.diarizer.out_dir)
    
    # Visualize nếu có RTTM output
    pred_rttm_dir = os.path.join(cfg.diarizer.out_dir, 'pred_rttms')
    if os.path.exists(pred_rttm_dir):
        rttm_files = [f for f in os.listdir(pred_rttm_dir) if f.endswith('.rttm')]
        if rttm_files:
            rttm_path = os.path.join(pred_rttm_dir, rttm_files[0])
            visualize_diarization(rttm_path, os.path.basename(audio_path))
    
    print(f"\n✓ ALL DONE! Results saved to: {os.path.abspath(cfg.diarizer.out_dir)}")
    
    return cfg.diarizer.out_dir


if __name__ == "__main__":
    # Configuration
    audio_file = "TestJ.wav"  # Thay bằng đường dẫn audio của bạn
    config_yaml = "diar_infer_config.yaml"
    output_dir = "diar_output"
    
    # Optional: Ground truth RTTM file để evaluate
    rttm_file = None  # Set đường dẫn nếu có ground truth
    
    # Optional: Số lượng speakers nếu biết trước (oracle mode)
    num_speakers = None  # Set số nếu muốn dùng oracle_num_speakers
    
    # Run diarization
    diarize_audio(
        audio_path=audio_file,
        config_yaml=config_yaml,
        output_dir=output_dir,
        rttm_file=rttm_file,
        num_speakers=num_speakers
    )
