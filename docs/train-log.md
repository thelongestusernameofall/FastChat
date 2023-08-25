1. zh04-pt - LORA

lora = 'q_proj, v_proj, k_proj, o_proj'， batch_size = 1, 400条数据

`batch_size = 1, epochs = 6, 476s, loss: 5.3 `   
`batch_size = 16, epochs = 10, 236s, loss: 14 `

2. zh05-pt - LORA

lora = 'q_proj, v_proj'， batch_size = 1, 400条数据

`batch_size = 1, epochs = 6, 300s, loss: 5.3`    
`batch_size = 1, epochs = 10, 500s, loss: 4.11`
`batch_size = 1, epochs = 20, 500s, loss: 3.5`
`batch_size = 16, epochs = 10, 267s, loss: 13`

3. zh06-pt - LORA

LORA parameter: 'q_proj,', 'v_proj,', 'k_proj,', 'o_proj,', 'gate_proj,', 'down_proj,', 'up_proj'， 400条数据    

```commandline
batch_size = 1, epochs = 30, 1546s, loss: 3.5
```

4.  zh06-pt - Full

batch_size = 1, 400条数据,

```
    epochs = 3, 480s, loss: 0.3 - 0.4 续写效果不好    
    epochs = 4, 660s, loss: 0.1     
    epochs = 6, 978s, loss: 0.08    
```