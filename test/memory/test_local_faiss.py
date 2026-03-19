import sys
import shutil
from pathlib import Path
import numpy as np

# 将项目 src 目录加入环境变量以支持导包
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / "src"))

from memory.storage.local_faiss import LocalFaissDatabase

def test_local_faiss():
    # 使用一个专门的测试目录
    test_db_dir = project_root / "test" / "test_db_temp"
    if test_db_dir.exists():
        shutil.rmtree(test_db_dir)
        
    print("=== Testing LocalFaissDatabase ===")
    
    # 初始化数据库
    db = LocalFaissDatabase(
        namespace="test_user_01",
        database_root=str(test_db_dir)
    )
    
    # 模拟几个维度为 4 的嵌入向量，方便预测余弦相似度的结果
    # 向量1: [1,0,0,0], 向量2: [0,1,0,0], 向量3: [0,0,1,0]
    emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    emb3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    
    print("\n1. 测试添加记忆 (Add Memories)...")
    id1 = db.add("记忆1: 我昨天吃了一个苹果", "session_1-turn_1", "2024/01/01 10:00", {"topic": "food"}, embedding=emb1)
    id2 = db.add("记忆2: 今天天气真不错，天空很蓝", "session_1-turn_2", "2024/01/02 10:00", {"topic": "weather"}, embedding=emb2)
    id3 = db.add("记忆3: 我正在学习机器学习", "session_2-turn_1", "2024/01/03 10:00", {"topic": "study"}, embedding=emb3)
    print(f"成功添加三条记录! IDs: \n- {id1}\n- {id2}\n- {id3}")
    
    print("\n2. 测试向量检索 (Search)...")
    # 我们用一个与 emb2 非常相近的向量 [0.1, 0.9, 0.0, 0.0] 来检索
    query_emb = np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32)
    results = db.search(query_emb, top_k=2)
    for idx, res in enumerate(results):
        print(f"检索排名 {idx+1}: {res.text} (得分: {res.score:.4f})")
    assert "天空很蓝" in results[0].text, "检索结果与预期不符！最相近的应该是记忆2。"

    print("\n3. 测试记忆更新 (Update Memory)...")
    # 稍微修改向量3
    new_emb3 = np.array([0.0, 0.0, 0.9, 0.1], dtype=np.float32)
    db.update_memory(
        id3, 
        new_text="记忆3 (更新): 我正在深入学习机器学习及其工程实践",
        metadata_updates={"updated": True},
        new_embedding=new_emb3
    )
    # 验证更新是否成功
    print("更新完成。检查全量记忆内容...")
    
    print("\n4. 测试全量获取 & 时间排序 (List All Memories)...")
    all_mems = db.list_all_memories(sort_by_time=True, descending=True)
    # 降序应该从 01/03 排到 01/01
    for mem in all_mems:
        print(f"- 时间: {mem.time} | 来源: {mem.source_index} | 文本: {mem.text} | 元信息: {mem.metadata}")
    assert "深入学习" in all_mems[0].text, "全量记忆排序或更新功能出错！"
    assert all_mems[0].time == "2024/01/03 10:00", "时间降序排序错误！"

    print("\n5. 测试持久化和懒加载 (Test Persistence & Lazy Load)...")
    # 重新实例化一个指向同一路径的连接
    db_reloaded = LocalFaissDatabase(
        namespace="test_user_01",
        database_root=str(test_db_dir)
    )
    # 强制它从硬盘加载
    all_mems_reloaded = db_reloaded.list_all_memories()
    print(f"从硬盘恢复的数据库中有 {len(all_mems_reloaded)} 条记录。")
    assert len(all_mems_reloaded) == 3, "持久化恢复失败，记录丢失！"
    
    print("\n6. 测试记忆删除 (Delete Memory)...")
    db_reloaded.delete(id1)
    all_mems_deleted = db_reloaded.list_all_memories()
    print(f"删除记录1后，数据库剩余 {len(all_mems_deleted)} 条记录。")
    assert len(all_mems_deleted) == 2, "删除功能失败！"
    
    # 最重要的一点：查下删掉向量后检索还能不能查准
    res_after_del = db_reloaded.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=1)
    # 因为 [1,0,0,0] 已经被删，现在的 top_k 应该不再是记忆1
    print(f"检索已删除向量 [1,0,0,0] 返回的最相近的项变成了：{res_after_del[0].text}")
    assert id1 != res_after_del[0].memory_id, "删除不彻底，向量表中还有残留！"

    print("\n==== 所有的增删改查测试通过! ====")
    # 扫尾清理
    # shutil.rmtree(test_db_dir)

if __name__ == "__main__":
    test_local_faiss()
