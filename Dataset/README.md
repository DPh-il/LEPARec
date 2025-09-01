@inproceedings{recbole[1.2.1],
  author    = {Lanling Xu and Zhen Tian and Gaowei Zhang and Junjie Zhang and Lei Wang and Bowen Zheng and Yifan Li and Jiakai Tang and Zeyu Zhang and Yupeng Hou and Xingyu Pan and Wayne Xin Zhao and Xu Chen and Ji{-}Rong Wen},
  title     = {Towards a More User-Friendly and Easy-to-Use Benchmark Library for Recommender Systems},
  booktitle = {{SIGIR}},
  pages     = {2837--2847},
  publisher = {{ACM}},
  year      = {2023}
}

Raw datasets information

| SN   | Dataset            | \#Items    | \#Linked\-Items | \#Users   | \#Interactions |
| ---- | ------------------ | ---------- | --------------- | --------- | -------------- |
| 1    | MovieLens          | 27,278     | 25,503          | 138,493   | 20,000,263     |
| 2    | Amazon\-book       | 2,370,605  | 108,515         | 8,026,324 | 22,507,155     |
| 3    | LFM\-1b \(tracks\) | 31,634,450 | 1,254,923       | 120,322   | 319,951,294    |

After filtering by 5-core (And filter out the tracks that are listened to less than 10 times in LFM-1b)

| SN   | Dataset            | \#Items | \#Linked\-Items | \#Users | \#Interactions |
| ---- | ------------------ | ------- | --------------- | ------- | -------------- |
| 1    | MovieLens          | 18,345  | 18,057          | 138,493 | 19,984,024     |
| 2    | Amazon\-book       | 367,982 | 34,476          | 603,668 | 8,898,041      |
| 3    | LFM\-1b \(tracks\) | 615,823 | 337,349         | 79,133  | 15,765,756     |