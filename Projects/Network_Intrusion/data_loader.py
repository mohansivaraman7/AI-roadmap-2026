"""
Data loading and preprocessing for the NSL-KDD network intrusion dataset.

NSL-KDD is a benchmark dataset for network intrusion detection systems (NIDS).
Each row is a single network connection record (the kind of thing a firewall,
router, or IDS sensor would log) described by 41 features, plus a label saying
whether the connection was 'normal' or a specific attack type.
"""

import pandas as pd
import numpy as np

# Official NSL-KDD column names (41 features + label)
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

# Mapping from the ~22-39 fine-grained attack names in NSL-KDD to the 4
# standard attack categories used in most published NSL-KDD research.
ATTACK_CATEGORY_MAP = {
    "normal": "normal",
    # DoS - Denial of Service
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "apache2": "dos", "udpstorm": "dos", "processtable": "dos",
    "mailbomb": "dos", "worm": "dos",
    # Probe - surveillance/scanning
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L - Remote to Local (unauthorized access from a remote machine)
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l", "snmpgetattack": "r2l",
    "httptunnel": "r2l", "sendmail": "r2l", "named": "r2l",
    # U2R - User to Root (privilege escalation)
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}


def load_dataset(train_path: str, test_path: str):
    """Load train/test CSVs, assign column names, and clean up NSL-KDD quirks."""
    train_df = pd.read_csv(train_path, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, names=COLUMN_NAMES)

    # Some NSL-KDD distributions carry an extra trailing 'difficulty' column
    # with no header - drop any unnamed/all-numeric extra column if present.
    for df in (train_df, test_df):
        if df.shape[1] > len(COLUMN_NAMES):
            df.drop(df.columns[len(COLUMN_NAMES):], axis=1, inplace=True)

    # Clean label strings (strip whitespace / trailing dots some versions use)
    train_df["label"] = train_df["label"].astype(str).str.strip().str.rstrip(".")
    test_df["label"] = test_df["label"].astype(str).str.strip().str.rstrip(".")

    # Map fine-grained attack labels -> {normal, dos, probe, r2l, u2r}
    train_df["attack_category"] = train_df["label"].map(ATTACK_CATEGORY_MAP).fillna("unknown")
    test_df["attack_category"] = test_df["label"].map(ATTACK_CATEGORY_MAP).fillna("unknown")

    # Binary label: 0 = normal traffic, 1 = attack traffic
    train_df["binary_label"] = (train_df["attack_category"] != "normal").astype(int)
    test_df["binary_label"] = (test_df["attack_category"] != "normal").astype(int)

    return train_df, test_df


if __name__ == "__main__":
    tr, te = load_dataset("data/NSL_KDD_Train.csv", "data/NSL_KDD_Test.csv")
    print("Train shape:", tr.shape)
    print("Test shape:", te.shape)
    print("\nTrain attack category counts:\n", tr["attack_category"].value_counts())
    print("\nTest attack category counts:\n", te["attack_category"].value_counts())
