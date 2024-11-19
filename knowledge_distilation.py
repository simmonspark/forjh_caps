import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, teacher_temp=2.0, alpha=0.5):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()  # Binary segmentation에 적합한 손실 함수

    def forward(self, student_output, teacher_output, ground_truth):
        # Distillation Loss
        soft_teacher = torch.sigmoid(teacher_output / self.teacher_temp)
        soft_student = torch.sigmoid(student_output / self.teacher_temp)
        kd_loss = F.binary_cross_entropy(soft_student, soft_teacher)

        # Binary Cross Entropy Loss
        ce_loss = self.bce_loss(student_output, ground_truth)

        # Combined Loss
        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss
