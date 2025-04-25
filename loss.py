import torch
import torch.nn.functional as F

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    NT-Xent Loss 계산 (Normalized Temperature-scaled Cross Entropy Loss)
    가까이 있는 샘플끼리 유사도 높게, 멀리 있는 샘플끼리 유사도 낮게

    Args:
        z_i (torch.Tensor): 첫 번째 뷰의 임베딩 (B, D)
        z_j (torch.Tensor): 두 번째 뷰의 임베딩 (B, D)
        temperature (float): 온도 스케일링 파라미터

    Returns:
        torch.Tensor: NT-Xent Loss 값
    """
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # 코사인 유사도 계산
    sim_matrix = sim_matrix / temperature

    # 각 샘플의 양의 인덱스를 지정
    # 예를 들어, 0~(B-1) 샘플의 양은 인덱스 B~(2B-1), 
    # 그리고 B~(2B-1) 샘플의 양은 인덱스 0~(B-1)이 됨
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)  # (2B,)

    # 자기 자신과의 유사도를 마스킹(너무 큰 음수 값 대신 -1e9 사용)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix = sim_matrix.masked_fill(mask, -1e9)

    # Cross Entropy Loss 계산
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def nt_xent_loss_with_reweighting(z_i, z_j, temperature=0.5, similarity_threshold=0.9):
    """
    NT-Xent Loss 계산 (Normalized Temperature-scaled Cross Entropy Loss)
    가까이 있는 샘플끼리 유사도 높게, 멀리 있는 샘플끼리 유사도 낮게

    Args:
        z_i (torch.Tensor): 첫 번째 뷰의 임베딩 (B, D)
        z_j (torch.Tensor): 두 번째 뷰의 임베딩 (B, D)
        temperature (float): 온도 스케일링 파라미터

    Returns:
        torch.Tensor: NT-Xent Loss 값
    """
    batch_size = z_i.size(0)
    device = z_i.device

    # 정규화된 임베딩
    z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
    z = F.normalize(z, dim=1)

    # 코사인 유사도 행렬 [2B, 2B]
    sim_matrix = torch.matmul(z, z.T) / temperature

    # 자기 자신은 유사도 제거
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    sim_matrix.masked_fill_(mask, float('-inf'))

    # positive 인덱스 설정
    pos_indices = torch.arange(batch_size, device=device)
    labels = torch.cat([pos_indices + batch_size, pos_indices], dim=0)  # (2B,)

    # Positive 쌍 마스크 설정 (이들은 항상 weight = 1이어야 함)
    positive_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
    for i in range(batch_size):
        positive_mask[i, i + batch_size] = True
        positive_mask[i + batch_size, i] = True
        
    # reweighting mask 생성
    with torch.no_grad():
        fn_mask = (sim_matrix > similarity_threshold) & (~positive_mask) & (~mask)
        weights = torch.ones_like(sim_matrix)
        weights[fn_mask] = 0.7  # false negative 추정 쌍의 weight 낮춤
        # positive pair는 항상 weight = 1로 명시적으로 보장   
        weights[positive_mask] = 1.0

    # softmax 계산을 위한 weighted sim matrix
    logits = sim_matrix * weights

    # cross_entropy를 위한 softmax 대상 정렬
    loss = F.cross_entropy(logits, labels)
    return loss