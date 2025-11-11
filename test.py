from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    'saves/glm4v_cubing_activitynet_test/checkpoint-200',
    trust_remote_code=True
)

print("模型所有模块：")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")
    
# 统计参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")