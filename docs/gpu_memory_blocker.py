import subprocess
import time
import torch
import os

def get_free_memory_nvidia_smi(device_id=0):
    """通过 nvidia-smi 查询显卡的空闲显存（单位：GB）"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi 查询失败: {result.stderr}")

    free_memory_list = result.stdout.strip().split('\n')
    free_memory_mb = int(free_memory_list[device_id])
    free_memory_gb = free_memory_mb / 1024.0
    return free_memory_gb

def allocate_dynamic_memory(device_id=0, keep_gb=3):
    """占用 (当前空闲 - 保留固定大小) 的显存"""
    with torch.cuda.device(device_id):
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device_id)

        # 实时获取最新的空闲显存
        current_free_gb = get_free_memory_nvidia_smi(device_id)
        
        # 保留固定大小的显存，尽可能占用剩余显存
        alloc_gb = current_free_gb - keep_gb

        if alloc_gb <= 0:
            print(f"当前空闲 {current_free_gb:.2f} GB，保留 {keep_gb:.2f} GB 后无须占用。")
            return None

        print(f"准备占用 {alloc_gb:.2f} GB 显存，保留 {keep_gb:.2f} GB 显存。")
        num_elements = int(alloc_gb * (1024**3) // 4)  # float32占4bytes
        tensor = torch.empty(num_elements, dtype=torch.float32, device=f'cuda:{device_id}')
        return tensor

def simulate_computation(device_id=0, duration_seconds=1, intensity=0.02):
    """模拟GPU计算，生成一个小的矩阵运算负载来控制GPU的计算量"""
    with torch.cuda.device(device_id):
        matrix_size = int(1024 * intensity)  # 控制规模
        a = torch.randn(matrix_size, matrix_size, device=f'cuda:{device_id}')
        b = torch.randn(matrix_size, matrix_size, device=f'cuda:{device_id}')
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            c = torch.mm(a, b)  # 矩阵乘法，模拟计算负载
            torch.cuda.synchronize()

def gpu_has_other_process(device_id=0):
    """检测GPU上是否存在其他进程（除自己以外）"""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits", f"--id={device_id}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi 查询进程失败: {result.stderr}")

    pids = result.stdout.strip().split('\n')
    pids = [int(pid) for pid in pids if pid.strip()]

    my_pid = os.getpid()
    other_pids = [pid for pid in pids if pid != my_pid]

    if other_pids:
        print(f"检测到GPU{device_id}上有其他进程存在: {other_pids}")
        return True
    else:
        return False

if __name__ == "__main__":
    device_ids = [6]  # 目标GPU编号列表
    keep_gb = 3  # 保留空闲显存（GB）

    allocated_tensors = {}  # 存放所有占用的 tensor，按GPU划分

    try:
        while True:
            for device_id in device_ids:
                free_mem = get_free_memory_nvidia_smi(device_id)
                print(f"nvidia-smi检测到GPU{device_id}的空闲显存: {free_mem:.2f} GB")

                if free_mem > 0.1:  # 超过100MB才占用，防止波动
                    tensor = allocate_dynamic_memory(device_id, keep_gb)
                    if tensor is not None:
                        if device_id not in allocated_tensors:
                            allocated_tensors[device_id] = []
                        allocated_tensors[device_id].append(tensor)
                        total_allocated_gb = sum(tensor.numel() * tensor.element_size() for tensor in allocated_tensors[device_id]) / (1024**3)
                        print(f"GPU{device_id}已占用显存 {total_allocated_gb:.2f} GB，保留 {keep_gb} GB 空闲显存。")

                    # 判断是否要进行计算
                    if not gpu_has_other_process(device_id):
                        print(f"GPU{device_id}上没有其他程序，开始模拟计算负载。")
                        simulate_computation(device_id, duration_seconds=1, intensity=0.02)
                    else:
                        print(f"GPU{device_id}上有其他程序，跳过计算，节省算力。")

                else:
                    print(f"GPU{device_id}空闲显存太小，无需占用。")

            time.sleep(10)  # 每10秒检测一次

    except KeyboardInterrupt:
        print("手动中断，程序退出。")
