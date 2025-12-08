"""
Test NeMo Speaker Diarization - Improved Approach with Transcription
Áp dụng đầy đủ từ nemo_diarization.py:
- Chia audio thành segments (sliding window)
- Extract speaker embeddings cho từng segment
- Cluster embeddings để tìm speakers
- Transcribe audio với Whisper ASR
- Tự implement speaker matching với embeddings
"""
import os
import torch
import numpy as np
import soundfile as sf
import whisper
from typing import Union, Optional, Dict, List, Tuple
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel


class SimpleSpeakerDiarization:
    """
    Simple Speaker Diarization using NeMo embeddings with segmentation
    
    Approach (from nemo_diarization.py):
    1. Chia audio thành segments (sliding window: 1.5s window, 0.75s shift)
    2. Extract speaker embedding cho từng segment
    3. Cluster các embeddings để tìm số speakers
    4. So sánh với speakers đã biết (nếu có memory)
    """
    
    def __init__(self, 
                 pretrained_speaker_model="titanet_large",
                 window_length_sec=1.5,  # Độ dài mỗi segment
                 shift_length_sec=0.75,  # Shift giữa các segments
                 similarity_threshold=0.7,
                 embedding_update_weight=0.3,
                 min_similarity_gap=0.15,
                 whisper_model_name="base"):
        """
        Initialize Simple Speaker Diarization with Transcription
        
        Parameters
        ----------
        pretrained_speaker_model : str
            NeMo pretrained model name (titanet_large, ecapa_tdnn, speakerverification_speakernet)
        window_length_sec : float
            Độ dài mỗi segment (giây)
        shift_length_sec : float
            Shift giữa các segments (giây)
        similarity_threshold : float
            Ngưỡng cosine similarity để match speaker (0.0 - 1.0)
        embedding_update_weight : float
            Trọng số cập nhật EMA embedding (0.0 - 1.0)
        min_similarity_gap : float
            Gap tối thiểu giữa best và second-best để match dù < threshold
        whisper_model_name : str
            Whisper model size (tiny, base, small, medium, large)
        """
        self.window_length_sec = window_length_sec
        self.shift_length_sec = shift_length_sec
        self.similarity_threshold = similarity_threshold
        self.embedding_update_weight = embedding_update_weight
        self.min_similarity_gap = min_similarity_gap
        self.sample_rate = 16000
        
        # Speaker memory
        self.speaker_memory = {}  # {speaker_id: embedding}
        self.speaker_counts = {}  # {speaker_id: count}
        self.speaker_clusters = {}  # {speaker_id: [embeddings]}
        self.next_speaker_id = 0
        self.max_cluster_size = 20
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🚀 Initializing NeMo Speaker Embedding Model...")
        print(f"   Device: {self.device}")
        print(f"   Model: {pretrained_speaker_model}")
        print(f"   Window: {window_length_sec}s, Shift: {shift_length_sec}s")
        print(f"   Similarity Threshold: {similarity_threshold}")
        print(f"   Min Gap: {min_similarity_gap}")
        
        # Load NeMo model
        self.speaker_model = EncDecSpeakerLabelModel.from_pretrained(
            model_name=pretrained_speaker_model
        )
        self.speaker_model.freeze()
        self.speaker_model.eval()
        self.speaker_model.to(self.device)
        
        print(f"✅ Model loaded successfully!")
        
        # Load Whisper ASR model
        print(f"\n🎯 Loading Whisper ASR model ({whisper_model_name})...")
        self.whisper_model = whisper.load_model(whisper_model_name, device=self.device)
        print(f"✅ Whisper model loaded successfully!")
    
    def segment_audio(self, audio: np.ndarray) -> List[Tuple[np.ndarray, float, float]]:
        """
        Chia audio thành segments với sliding window
        
        Parameters
        ----------
        audio : np.ndarray
            Audio array (float32, 16kHz)
            
        Returns
        -------
        segments : List[Tuple[np.ndarray, float, float]]
            List of (segment_audio, start_time, end_time)
        """
        window_samples = int(self.window_length_sec * self.sample_rate)
        shift_samples = int(self.shift_length_sec * self.sample_rate)
        
        segments = []
        audio_length = len(audio)
        
        start = 0
        while start < audio_length:
            end = min(start + window_samples, audio_length)
            segment = audio[start:end]
            
            # Chỉ lấy segment đủ dài (>= 0.5s)
            if len(segment) >= self.sample_rate * 0.5:
                start_time = start / self.sample_rate
                end_time = end / self.sample_rate
                segments.append((segment, start_time, end_time))
            
            start += shift_samples
            
            # Break nếu đã đến cuối
            if end >= audio_length:
                break
        
        return segments
    
    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract speaker embedding từ audio segment
        
        Parameters
        ----------
        audio : np.ndarray
            Audio array (float32, 16kHz)
            
        Returns
        -------
        embedding : np.ndarray
            Normalized speaker embedding vector
        """
        # Prepare input
        audio_length = audio.shape[0]
        audio_signal = torch.tensor(audio, device=self.device, dtype=torch.float32).unsqueeze(0)
        audio_signal_len = torch.tensor([audio_length], device=self.device)
        
        # Extract embedding
        with torch.no_grad():
            _, emb = self.speaker_model.forward(audio_signal, audio_signal_len)
            # emb shape: (batch, time, embedding_dim) -> squeeze to (embedding_dim,)
            emb = emb.squeeze(0).detach().cpu().numpy()
        
        # Normalize embedding
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        
        return emb_norm
    
    def extract_embeddings_from_segments(self, audio: np.ndarray) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Extract embeddings từ tất cả segments của audio
        
        Parameters
        ----------
        audio : np.ndarray
            Full audio array (float32, 16kHz)
            
        Returns
        -------
        embeddings : np.ndarray
            Array of embeddings (num_segments, embedding_dim)
        time_ranges : List[Tuple[float, float]]
            List of (start_time, end_time) cho mỗi segment
        """
        # Segment audio
        segments = self.segment_audio(audio)
        print(f"  🔪 Segmented into {len(segments)} segments")
        
        # Extract embeddings
        embeddings = []
        time_ranges = []
        
        for i, (segment_audio, start_time, end_time) in enumerate(segments):
            emb = self.extract_embedding(segment_audio)
            embeddings.append(emb)
            time_ranges.append((start_time, end_time))
        
        embeddings_array = np.array(embeddings)
        print(f"  📊 Extracted {len(embeddings)} embeddings, shape: {embeddings_array.shape}")
        
        return embeddings_array, time_ranges
    
    def cluster_embeddings(self, embeddings: np.ndarray, 
                          num_speakers: Optional[int] = None,
                          max_speakers: int = 8) -> np.ndarray:
        """
        Cluster embeddings để tìm speakers
        
        Parameters
        ----------
        embeddings : np.ndarray
            Array of embeddings (num_segments, embedding_dim)
        num_speakers : int, optional
            Số speakers cố định (nếu biết trước)
        max_speakers : int
            Số speakers tối đa
            
        Returns
        -------
        labels : np.ndarray
            Cluster labels cho mỗi embedding
        """
        if num_speakers is None:
            # Auto-detect số speakers bằng cách thử các giá trị khác nhau
            best_n = 1
            best_score = -np.inf
            
            for n in range(1, min(max_speakers + 1, len(embeddings))):
                clusterer = AgglomerativeClustering(
                    n_clusters=n,
                    metric='cosine',
                    linkage='average'
                )
                labels = clusterer.fit_predict(embeddings)
                
                # Tính silhouette score (đơn giản: inertia trung bình)
                if n > 1:
                    # Tính average distance within clusters
                    score = 0
                    for label in range(n):
                        cluster_embs = embeddings[labels == label]
                        if len(cluster_embs) > 1:
                            centroid = cluster_embs.mean(axis=0)
                            dists = cdist(cluster_embs, [centroid], metric='cosine').flatten()
                            score -= dists.mean()
                    
                    if score > best_score:
                        best_score = score
                        best_n = n
            
            num_speakers = best_n
            print(f"  🎯 Auto-detected {num_speakers} speakers (max tried: {min(max_speakers, len(embeddings))})")
        
        # Final clustering
        clusterer = AgglomerativeClustering(
            n_clusters=num_speakers,
            metric='cosine',
            linkage='average'
        )
        labels = clusterer.fit_predict(embeddings)
        
        return labels
    
    def transcribe_audio_segments(self, audio: np.ndarray, 
                                  speaker_labels: List[str],
                                  time_ranges: List[Tuple[float, float]]) -> Dict[str, List[Dict]]:
        """
        Transcribe audio cho từng speaker segment (merged consecutive segments)
        
        Parameters
        ----------
        audio : np.ndarray
            Full audio array (float32, 16kHz)
        speaker_labels : List[str]
            Speaker labels cho mỗi segment
        time_ranges : List[Tuple[float, float]]
            Time ranges cho mỗi segment
            
        Returns
        -------
        transcripts : Dict[str, List[Dict]]
            {
                'SPEAKER_00': [
                    {'start': 0.0, 'end': 3.5, 'text': 'Hello world'},
                    ...
                ],
                ...
            }
        """
        print(f"\n🎤 Transcribing audio segments...")
        
        # Merge consecutive segments from same speaker
        merged_segments = []
        current_speaker = None
        current_start = None
        current_end = None
        
        for speaker_id, (start, end) in zip(speaker_labels, time_ranges):
            if speaker_id != current_speaker:
                if current_speaker is not None:
                    merged_segments.append((current_speaker, current_start, current_end))
                current_speaker = speaker_id
                current_start = start
                current_end = end
            else:
                current_end = end
        
        # Add last segment
        if current_speaker is not None:
            merged_segments.append((current_speaker, current_start, current_end))
        
        print(f"  Found {len(merged_segments)} merged speaker segments")
        
        # Transcribe each merged segment
        transcripts = {}
        
        for i, (speaker_id, start_time, end_time) in enumerate(merged_segments):
            # Extract audio segment
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            segment_audio = audio[start_sample:end_sample]
            
            # Skip very short segments (< 0.5s)
            duration = end_time - start_time
            if duration < 0.5:
                continue
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                segment_audio,
                language='ja',  # hoặc None để auto-detect
                fp16=False if self.device == 'cpu' else True
            )
            
            text = result['text'].strip()
            
            # Add to transcripts
            if speaker_id not in transcripts:
                transcripts[speaker_id] = []
            
            transcripts[speaker_id].append({
                'start': start_time,
                'end': end_time,
                'duration': duration,
                'text': text
            })
            
            print(f"  [{i+1}/{len(merged_segments)}] {speaker_id} ({start_time:.2f}s-{end_time:.2f}s): {text[:50]}...")
        
        print(f"  ✅ Transcription completed")
        
        return transcripts
    
    def process_audio(self, audio_path_or_array: Union[str, np.ndarray],
                     num_speakers: Optional[int] = None,
                     max_speakers: int = 8,
                     use_memory: bool = False,
                     transcribe: bool = True) -> Dict:
        """
        Process audio và trả về kết quả diarization với transcription
        
        Parameters
        ----------
        audio_path_or_array : str or np.ndarray
            Path to audio file or audio array
        num_speakers : int, optional
            Số speakers cố định (nếu biết trước)
        max_speakers : int
            Số speakers tối đa
        use_memory : bool
            Có sử dụng speaker memory để match với speakers đã biết không
        transcribe : bool
            Có transcribe audio không
            
        Returns
        -------
        result : dict
            {
                'num_speakers': int,
                'speaker_labels': List[str],  # Speaker cho mỗi segment
                'time_ranges': List[Tuple[float, float]],  # Time range của mỗi segment
                'embeddings': np.ndarray,  # Embeddings của mỗi segment
                'speaker_times': Dict[str, float],  # Tổng thời gian của mỗi speaker
                'transcripts': Dict[str, List[Dict]]  # Transcripts cho mỗi speaker
            }
        """
        print(f"\n{'='*80}")
        print(f"🎤 Processing audio...")
        
        # Load audio
        if isinstance(audio_path_or_array, str):
            audio, sr = sf.read(audio_path_or_array)
            if sr != 16000:
                # Resample to 16kHz
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            audio = audio.astype(np.float32)
        else:
            audio = audio_path_or_array.astype(np.float32)
        
        duration = len(audio) / self.sample_rate
        print(f"  Audio duration: {duration:.2f}s")
        
        # Extract embeddings from segments
        embeddings, time_ranges = self.extract_embeddings_from_segments(audio)
        
        # Cluster embeddings
        cluster_labels = self.cluster_embeddings(embeddings, num_speakers, max_speakers)
        
        # Convert cluster labels to speaker IDs
        unique_clusters = np.unique(cluster_labels)
        num_detected_speakers = len(unique_clusters)
        print(f"  👥 Detected {num_detected_speakers} speakers in audio")
        
        # Assign speaker IDs
        speaker_labels = []
        cluster_to_speaker = {}  # Map cluster label -> SPEAKER_ID
        
        for cluster_label in cluster_labels:
            if cluster_label not in cluster_to_speaker:
                if use_memory and len(self.speaker_memory) > 0:
                    # Match với speakers đã biết
                    segment_idx = len(speaker_labels)
                    segment_emb = embeddings[segment_idx]
                    matched_speaker_id = self._match_with_memory(segment_emb)
                    cluster_to_speaker[cluster_label] = matched_speaker_id
                else:
                    # Tạo speaker ID mới
                    speaker_id = f"SPEAKER_{self.next_speaker_id:02d}"
                    self.next_speaker_id += 1
                    cluster_to_speaker[cluster_label] = speaker_id
            
            speaker_labels.append(cluster_to_speaker[cluster_label])
        
        # Update speaker memory
        if use_memory:
            for speaker_id in cluster_to_speaker.values():
                if speaker_id not in self.speaker_memory:
                    # Initialize với centroid của cluster
                    speaker_embeddings = embeddings[[
                        i for i, label in enumerate(cluster_labels) 
                        if cluster_to_speaker[label] == speaker_id
                    ]]
                    centroid = speaker_embeddings.mean(axis=0)
                    centroid = centroid / np.linalg.norm(centroid)
                    self.speaker_memory[speaker_id] = centroid
                    self.speaker_clusters[speaker_id] = [centroid]
                    self.speaker_counts[speaker_id] = 1
        
        # Calculate speaker times
        speaker_times = {}
        for speaker_id, (start_time, end_time) in zip(speaker_labels, time_ranges):
            duration_seg = end_time - start_time
            if speaker_id not in speaker_times:
                speaker_times[speaker_id] = 0
            speaker_times[speaker_id] += duration_seg
        
        # Transcribe audio segments
        transcripts = {}
        if transcribe:
            transcripts = self.transcribe_audio_segments(audio, speaker_labels, time_ranges)
        
        result = {
            'num_speakers': num_detected_speakers,
            'speaker_labels': speaker_labels,
            'time_ranges': time_ranges,
            'embeddings': embeddings,
            'speaker_times': speaker_times,
            'cluster_to_speaker': cluster_to_speaker,
            'transcripts': transcripts
        }
        
        print(f"{'='*80}\n")
        
        return result
    
    def _match_with_memory(self, new_embedding: np.ndarray) -> str:
        """Match embedding với speakers trong memory (simplified version)"""
        if len(self.speaker_memory) == 0:
            speaker_id = f"SPEAKER_{self.next_speaker_id:02d}"
            self.next_speaker_id += 1
            return speaker_id
        
        # Tính similarity
        speaker_ids = list(self.speaker_memory.keys())
        known_embeddings = np.array([self.speaker_memory[sid] for sid in speaker_ids])
        similarities = 1 - cdist([new_embedding], known_embeddings, metric='cosine')[0]
        
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= self.similarity_threshold:
            return speaker_ids[best_idx]
        else:
            # Create new
            speaker_id = f"SPEAKER_{self.next_speaker_id:02d}"
            self.next_speaker_id += 1
            return speaker_id 
    
    def get_speaker_info(self) -> Dict:
        """Lấy thông tin về speakers đã biết"""
        return {
            'speakers': list(self.speaker_memory.keys()),
            'speaker_counts': self.speaker_counts.copy(),
            'cluster_sizes': {sid: len(self.speaker_clusters[sid]) for sid in self.speaker_memory.keys()},
            'num_speakers': len(self.speaker_memory)
        }
    
    def reset(self):
        """Reset toàn bộ speaker memory"""
        self.speaker_memory.clear()
        self.speaker_counts.clear()
        self.speaker_clusters.clear()
        self.next_speaker_id = 0
        print("🔄 Reset all speaker memory")


# ============ EXAMPLE USAGE ============
if __name__ == "__main__":
    import sys
    
    # Initialize diarization with Whisper
    diarizer = SimpleSpeakerDiarization(
        pretrained_speaker_model="titanet_large",
        window_length_sec=1.5,  # Mỗi segment 1.5s
        shift_length_sec=0.75,  # Overlap 50%
        similarity_threshold=0.7,
        embedding_update_weight=0.3,
        min_similarity_gap=0.3,
        whisper_model_name="base"  # tiny, base, small, medium, large
    )
    
    # Test với audio file
    audio_file = "TestJ.wav"
    
    print("\n" + "="*100)
    print("TESTING SPEAKER DIARIZATION WITH SEGMENTATION")
    print("="*100)
    
    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
    else:
        print(f"\n\n{'#'*100}")
        print(f"# PROCESSING: {audio_file}")
        print(f"{'#'*100}")
        
        # Process audio with transcription
        result = diarizer.process_audio(
            audio_file,
            num_speakers=None,  # Auto-detect
            max_speakers=8,
            use_memory=False,
            transcribe=True  # Enable transcription
        )
        
        # Display results
        print(f"\n📊 DIARIZATION RESULTS:")
        print(f"   Number of speakers detected: {result['num_speakers']}")
        print(f"   Total segments: {len(result['speaker_labels'])}")
        
        print(f"\n⏱️  SPEAKER TIME DISTRIBUTION:")
        for speaker_id, total_time in sorted(result['speaker_times'].items()):
            print(f"   {speaker_id}: {total_time:.2f}s ({total_time/sum(result['speaker_times'].values())*100:.1f}%)")
        
        # Display transcripts
        if result['transcripts']:
            print(f"\n📝 TRANSCRIPTS BY SPEAKER:")
            for speaker_id in sorted(result['transcripts'].keys()):
                print(f"\n   {speaker_id}:")
                print(f"   {'─'*80}")
                for segment in result['transcripts'][speaker_id]:
                    start = segment['start']
                    end = segment['end']
                    text = segment['text']
                    print(f"   [{start:6.2f}s - {end:6.2f}s] {text}")
        
        print(f"\n📝 SEGMENT-BY-SEGMENT BREAKDOWN:")
        print(f"   {'Segment':<8} {'Start':<8} {'End':<8} {'Duration':<10} {'Speaker':<12}")
        print(f"   {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*12}")
        for i, (speaker_id, (start, end)) in enumerate(zip(result['speaker_labels'], result['time_ranges'])):
            duration = end - start
            print(f"   {i+1:<8} {start:<8.2f} {end:<8.2f} {duration:<10.2f} {speaker_id:<12}")
        
        # Merge consecutive segments from same speaker
        print(f"\n🔗 MERGED SPEAKER SEGMENTS:")
        merged_segments = []
        current_speaker = None
        current_start = None
        current_end = None
        
        for speaker_id, (start, end) in zip(result['speaker_labels'], result['time_ranges']):
            if speaker_id != current_speaker:
                if current_speaker is not None:
                    merged_segments.append((current_speaker, current_start, current_end))
                current_speaker = speaker_id
                current_start = start
                current_end = end
            else:
                current_end = end
        
        # Add last segment
        if current_speaker is not None:
            merged_segments.append((current_speaker, current_start, current_end))
        
        print(f"   {'Speaker':<12} {'Start':<8} {'End':<8} {'Duration':<10}")
        print(f"   {'-'*12} {'-'*8} {'-'*8} {'-'*10}")
        for speaker_id, start, end in merged_segments:
            duration = end - start
            print(f"   {speaker_id:<12} {start:<8.2f} {end:<8.2f} {duration:<10.2f}")
    
    print("\n" + "="*100)
    print("✅ TESTING COMPLETED")
    print("="*100)
