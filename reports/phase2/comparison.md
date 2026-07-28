# Phase 2 model comparison

## Overall (all folds, all horizons)

```
                       n  n_missing     mae    rmse  mape  rmse/mae
model
lightgbm            3360          0   272.5   375.7   2.0       1.4
seasonal_naive_168  3360          0   564.6   789.3   4.1       1.4
sarimax_fourier     3360          0   666.6   907.4   4.8       1.4
seasonal_naive_24   3360          0   673.4 1,022.1   4.9       1.5
entsoe_da_debiased  3360          0   903.0 1,248.1   6.7       1.4
persistence         3360          0 1,772.9 2,249.4  12.2       1.3
entsoe_day_ahead    3360          0 1,976.2 2,563.4  14.5       1.3
```

## MAE by horizon (1-24)

```
model    lightgbm  sarimax_fourier  seasonal_naive_168
horizon
1             157               70                 345
2             145              125                 336
3             150              181                 333
4             167              230                 334
5             214              383                 411
6             257              628                 516
7             271              836                 576
8             292              922                 594
9             320              886                 661
10            323              819                 721
11            331              792                 777
12            346              792                 797
13            379              819                 805
14            352              845                 812
15            335              867                 804
16            361              936                 758
17            395            1,045                 700
18            320              987                 610
19            284              842                 548
20            261              694                 497
21            245              638                 457
22            213              588                 415
23            212              542                 375
24            210              529                 367
```

## MAE by local hour (Europe/Amsterdam)

```
model       lightgbm  sarimax_fourier  seasonal_naive_168
hour_local
0                216              531                 371
1                187              255                 357
2                153              102                 350
3                146              154                 332
4                151              209                 323
5                177              256                 363
6                215              447                 440
7                273              792                 550
8                302              974                 593
9                328              945                 660
10               333              872                 726
11               339              825                 782
12               324              780                 781
13               359              803                 785
14               355              831                 806
15               342              839                 776
16               333              877                 742
17               395              965                 712
18               340              939                 655
19               305              944                 599
20               287              823                 541
21               252              685                 489
22               224              605                 430
23               204              546                 386
```

## Per-fold MAE distribution

```
                    mean   50%     90%     99%     max
model
lightgbm           272.5 223.8   488.9   730.1   961.5
sarimax_fourier    666.6 562.7 1,136.7 1,820.7 2,164.2
seasonal_naive_168 564.6 474.7 1,066.6 1,550.3 2,704.9
```

## Worst 10 LightGBM folds

```
     forecast_day        dow  fold_mae  naive168_mae
fold
124    2026-03-11  Wednesday     961.5       1,245.6
34     2024-12-16     Monday     773.7         272.4
112    2026-01-10   Saturday     662.0       1,006.1
110    2025-12-31  Wednesday     634.0         667.3
41     2025-01-20     Monday     633.3         796.5
25     2024-11-01     Friday     581.1       1,265.3
97     2025-10-27     Monday     560.8       1,056.6
111    2026-01-05     Monday     552.7       1,610.1
125    2026-03-16     Monday     538.9         498.7
101    2025-11-16     Sunday     530.6       1,047.7
```

## Head-to-head

```
lightgbm beats seasonal_naive_168 on 87.9% of 140 folds
```
