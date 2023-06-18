#!/bin/zsh
lr_ls=(1e-4 5e-4)
bs_ls=(32 64)
dr_ls=(0.95 0.9 0.75 0.6)

for lr in "${lr_ls[@]}";
  do
    for bs in "${bs_ls[@]}";
      do
        for dr in "${dr_ls[@]}";
          do
            python main_server.py --lr ${lr} --bs ${bs} --dropout_ratio ${dr}
          done
      done
  done
