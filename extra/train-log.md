

1. zh04-pt  - LORA    
lora = 'q_proj, v_proj, k_proj, o_proj'， batch_size = 1, epochs = 6, 400条数据    
    epochs = 6, 476s, loss:5.3
2. zh05-pt  - LORA    
lora = 'q_proj, v_proj'， batch_size = 1, 400条数据        
    epochs = 6, 300s, loss:5.3    
3. zh06-pt  - Full     
batch_size = 1, 400条数据,     
    epochs = 3, 480s, loss:0.3    
    epochs = 4, 660s, loss:0.1     
    epochs = 6, 978s, loss:0.08    