import sys
from pathlib import Path

# Add src to sys.path so we can import benchmark
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / "src"))

from benchmark import get_benchmark, MemoryEpisode

def test_locomo():
    print("=== Testing LoCoMo Benchmark ===")
    # 优先使用 raw_data，触发 LocomoBenchmark 的自动转换逻辑
    candidate_paths = [
        project_root / "data" / "raw_data" / "locomo10.json"
    ]
    file_path = next((p for p in candidate_paths if p.exists()), None)
    if file_path is None:
        print("File not found in candidates:")
        for p in candidate_paths:
            print(f"  - {p}")
        return
        
    benchmark = get_benchmark("locomo", str(file_path), lang="en")
    print(f"Total episodes loaded: {len(benchmark)}")
    
    if len(benchmark) > 0:
        episode: MemoryEpisode = benchmark.episodes[0]
        print(f"First Episode ID: {episode.history_name}")
        print(f"Number of sessions: {len(episode.sessions)}")
        print(f"Number of questions: {len(episode.qas)}")
        
        print("\nSample Session 0 (first 2 turns):")
        print(f"Date: {episode.sessions[0].session_date}")
        for turn in episode.sessions[0].turns[:2]:
            print(f"  {turn.speaker}: {turn.content}")
            
        print("\nSample QA 0:")
        print(f"Question: {episode.qas[0].question}")
        print(f"Answer: {episode.qas[0].answer}")
        print(f"Metadata: {episode.qas[0].metadata}")
        
    print("-" * 40)

def test_lme_oracle():
    print("\n=== Testing LongMemEval Benchmark (Oracle) ===")
    file_path = project_root / "data" / "raw_data" / "longmemeval_s_cleaned.json"
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
        
    benchmark = get_benchmark("lme_oracle", str(file_path), lang="en")
    print(f"Total episodes loaded: {len(benchmark)}")
    
    if len(benchmark) > 0:
        episode: MemoryEpisode = benchmark.episodes[0]
        print(f"First Episode ID: {episode.history_name}")
        print(f"Number of sessions: {len(episode.sessions)}")
        print(f"Number of questions: {len(episode.qas)}")
        
        if episode.sessions:
            print("\nSample Session 0 (first 2 turns):")
            print(f"Date: {episode.sessions[0].session_date}")
            for turn in episode.sessions[0].turns[:2]:
                print(f"  {turn.speaker}: {turn.content}")
                
        print("\nSample QA 0:")
        print(f"Question: {episode.qas[0].question}")
        print(f"Answer: {episode.qas[0].answer}")
        print(f"Options: {episode.qas[0].options}")
        
    print("-" * 40)

def test_lmb_event():
    print("\n=== Testing LifeMemBench Event ===")
    file_path = project_root / "data" / "preprocessed" / "LifeMemBench_event.json"
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
        
    benchmark = get_benchmark("lmb_event", str(file_path), lang="zh")
    print(f"Total episodes loaded: {len(benchmark)}")
    
    if len(benchmark) > 0:
        episode: MemoryEpisode = benchmark.episodes[0]
        print(f"First Episode ID: {episode.history_name}")
        print(f"Number of sessions: {len(episode.sessions)}")
        print(f"Number of questions: {len(episode.qas)}")
        
        print("\nSample Session 0 (first 2 turns):")
        print(f"Date: {episode.sessions[0].session_date}")
        for turn in episode.sessions[0].turns[:2]:
            print(f"  {turn.speaker}: {turn.content}")
            
        print("\nSample QA 0:")
        print(f"Question: {episode.qas[0].question}")
        print(f"Answer: {episode.qas[0].answer}")
        print(f"Options: {episode.qas[0].options}")
        print(f"Golden Option (from metadata): {episode.qas[0].metadata.get('golden_option')}")
        
    print("-" * 40)

def main():
    print("Testing Benchmark Loaders...")
    test_locomo()
    test_lme_oracle()
    test_lmb_event()
    print("Done testing.")

if __name__ == "__main__":
    main()
