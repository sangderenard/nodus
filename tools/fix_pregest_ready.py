fp = r'c:\dev\Powershell\nodus\toys_to_survive_development\wav_config_transformer_pipeline.py'
with open(fp, 'r', encoding='utf-8') as f:
    c = f.read()

old = 'pregestation_gate_streak >= int(gate_config.get("pregestation_maintain_rounds", 1))'
new = 'pregestation_sub_round >= int(gate_config.get("pregestation_sub_rounds", 1))'
count_before = c.count(old)
c2 = c.replace(old, new)
count_after_old = c2.count(old)
count_after_new = c2.count(new)
print(f"Replaced {count_before} occurrences. Remaining old: {count_after_old}. New count: {count_after_new}")

with open(fp, 'w', encoding='utf-8') as f:
    f.write(c2)
print("Done")
