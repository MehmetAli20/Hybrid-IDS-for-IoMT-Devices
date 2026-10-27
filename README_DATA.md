# Data Setup

The dataset is not included in this repository. Download the **CIC IoMT 2024** CSV files from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html) and place them like this:

```
data/
├── train/   # 51 *_train.pcap.csv files
└── test/    # 21 *_test.pcap.csv files
```

- Keep the original file names: each row's label comes from its file name (for example `TCP_IP-DDoS-UDP3_train.pcap.csv` → `DDoS-UDP`).
- The whole `data/` folder is ignored by git.
- On the first run the scripts read every CSV once and cache them in `data/cache/` (about 1.5 GB). Delete that folder after adding or replacing CSV files.

## Training files (`data/train/`, 51 files)

- ARP_Spoofing_train.pcap.csv
- Benign_train.pcap.csv
- MQTT-DDoS-Connect_Flood_train.pcap.csv
- MQTT-DDoS-Publish_Flood_train.pcap.csv
- MQTT-DoS-Connect_Flood_train.pcap.csv
- MQTT-DoS-Publish_Flood_train.pcap.csv
- MQTT-Malformed_Data_train.pcap.csv
- Recon-OS_Scan_train.pcap.csv
- Recon-Ping_Sweep_train.pcap.csv
- Recon-Port_Scan_train.pcap.csv
- Recon-VulScan_train.pcap.csv
- TCP_IP-DDoS-ICMP1_train.pcap.csv … TCP_IP-DDoS-ICMP8_train.pcap.csv (8 files)
- TCP_IP-DDoS-SYN1_train.pcap.csv … TCP_IP-DDoS-SYN4_train.pcap.csv (4 files)
- TCP_IP-DDoS-TCP1_train.pcap.csv … TCP_IP-DDoS-TCP4_train.pcap.csv (4 files)
- TCP_IP-DDoS-UDP1_train.pcap.csv … TCP_IP-DDoS-UDP8_train.pcap.csv (8 files)
- TCP_IP-DoS-ICMP1_train.pcap.csv … TCP_IP-DoS-ICMP4_train.pcap.csv (4 files)
- TCP_IP-DoS-SYN1_train.pcap.csv … TCP_IP-DoS-SYN4_train.pcap.csv (4 files)
- TCP_IP-DoS-TCP1_train.pcap.csv … TCP_IP-DoS-TCP4_train.pcap.csv (4 files)
- TCP_IP-DoS-UDP1_train.pcap.csv … TCP_IP-DoS-UDP4_train.pcap.csv (4 files)

## Test files (`data/test/`, 21 files)

- ARP_Spoofing_test.pcap.csv
- Benign_test.pcap.csv
- MQTT-DDoS-Connect_Flood_test.pcap.csv
- MQTT-DDoS-Publish_Flood_test.pcap.csv
- MQTT-DoS-Connect_Flood_test.pcap.csv
- MQTT-DoS-Publish_Flood_test.pcap.csv
- MQTT-Malformed_Data_test.pcap.csv
- Recon-OS_Scan_test.pcap.csv
- Recon-Ping_Sweep_test.pcap.csv
- Recon-Port_Scan_test.pcap.csv
- Recon-VulScan_test.pcap.csv
- TCP_IP-DDoS-ICMP1_test.pcap.csv
- TCP_IP-DDoS-ICMP2_test.pcap.csv
- TCP_IP-DDoS-SYN_test.pcap.csv
- TCP_IP-DDoS-TCP_test.pcap.csv
- TCP_IP-DDoS-UDP1_test.pcap.csv
- TCP_IP-DDoS-UDP2_test.pcap.csv
- TCP_IP-DoS-ICMP_test.pcap.csv
- TCP_IP-DoS-SYN_test.pcap.csv
- TCP_IP-DoS-TCP_test.pcap.csv
- TCP_IP-DoS-UDP_test.pcap.csv
