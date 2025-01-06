import pandas as pd
import numpy as np
import glob


def extract_ihs2018_dataset_weekly(specialty):
    dat_all = pd.read_csv("../data/2018_imputed_data_weekly.csv")
    # dat_all = pd.read_csv(
    #     "/home/jitwang/dynamic_mediation_project/python_version/real_data_analysis/data/2018_imputed_data_weekly.csv"
    # )
    if specialty == "all":
        dat = dat_all
    else:
        dat = dat_all[dat_all["specialty"] == specialty].copy(deep=True)
    dat = dat.sort_values(by=["STUDY_PRTCPT_ID", "week"]).reset_index().copy(deep=True)
    N = len(set(dat["STUDY_PRTCPT_ID"]))
    T = len(set(dat["week"]))
    # sqrt_step_count, sqrt_sleep_min, mood
    MDIM = 2
    M = np.zeros([N, T, MDIM], dtype=float)
    R = np.zeros([N, T], dtype=float)
    A = np.zeros([N, T], dtype=float)
    for id, i in zip(
        sorted(set(dat["STUDY_PRTCPT_ID"])), range(len(set(dat["STUDY_PRTCPT_ID"])))
    ):
        tmp = dat[dat["STUDY_PRTCPT_ID"] == id].reset_index()

        A[i, :] = tmp["msg_sent"].values

        R[i, :] = tmp["mood"].values

        M[i, :, :] = tmp[["step_count", "sleep_count"]].to_numpy()
    A = A.astype(int)

    return M, A, R


class IHS2018Dataset:
    def __init__(self, specialty, msg_type, reward_name, mediator_names) -> None:
        all_data = pd.read_csv(
            "../data/2018_imputed_data_weekly_hr.csv", na_filter=False
        )
        if specialty == "all":
            self.data = all_data
        else:
            self.data = all_data[all_data["specialty"] == specialty].copy(deep=True)
        assert msg_type in ["activity", "sleep", "mood", "all"]
        self.msg_type = msg_type
        assert reward_name in [
            "STEP_COUNT",
            "SLEEP_COUNT",
            "resting_hr",
            "rmssd",
            "MOOD",
        ]
        self.reward_name = reward_name
        assert set(mediator_names).issubset(
            set(["STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd", "MOOD"])
        )
        self.mediator_names = mediator_names

    def load_data(self):
        dat = (
            self.data.sort_values(by=["USERID", "study_week"])
            .reset_index()
            .copy(deep=True)
        )
        self.N = len(set(dat["USERID"]))
        self.T = len(set(dat["study_week"]))

        MDIM = len(self.mediator_names)
        M = np.zeros([self.N, self.T, MDIM], dtype=float)
        R = np.zeros([self.N, self.T], dtype=float)
        A = np.zeros([self.N, self.T], dtype=float)
        for id, i in zip(sorted(set(dat["USERID"])), range(len(set(dat["USERID"])))):
            tmp = dat[dat["USERID"] == id].reset_index()
            if self.msg_type == "all":
                A[i, :] = (tmp["week_category"].values != "None") + 0
            else:
                A[i, :] = (tmp["week_category"].values == self.msg_type) + 0

            R[i, :] = tmp[self.reward_name].values

            M[i, :, :] = tmp[self.mediator_names].to_numpy()
        A = A.astype(int)
        self.num_mediators = len(self.mediator_names)

        return M, A, R


class IHS2018DatasetMI:
    def __init__(self, specialty, msg_type, reward_name, mediator_names) -> None:
        dir_path = "../data/2018_weekly/*.*"
        self.data = {}
        for idx, file in enumerate(glob.glob(dir_path)):
            all_data = pd.read_csv(file, na_filter=False)
            if specialty == "all":
                self.data[idx] = all_data
            else:
                self.data[idx] = all_data[all_data["specialty"] == specialty].copy(
                    deep=True
                )
        assert msg_type in ["activity", "sleep", "mood", "all"]
        self.msg_type = msg_type
        assert reward_name in [
            "STEP_COUNT",
            "SLEEP_COUNT",
            "resting_hr",
            "rmssd",
            "MOOD",
        ]
        self.reward_name = reward_name
        assert set(mediator_names).issubset(
            set(["STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd", "MOOD"])
        )
        self.mediator_names = mediator_names
        self.num_mediators = len(self.mediator_names)
        self.N = len(set(self.data[0]["USERID"]))
        self.T = len(set(self.data[0]["study_week"]))
        self.num_imputations = len(self.data)

    def load_data(self, index):
        dat = (
            self.data[index]
            .sort_values(by=["USERID", "study_week"])
            .reset_index()
            .copy(deep=True)
        )
        # dat["rmssd"] = np.log(dat["rmssd"])
        # dat["MOOD"] = dat["MOOD"]

        MDIM = len(self.mediator_names)
        M = np.zeros([self.N, self.T, MDIM], dtype=float)
        R = np.zeros([self.N, self.T], dtype=float)
        A = np.zeros([self.N, self.T], dtype=float)
        for id, i in zip(sorted(set(dat["USERID"])), range(len(set(dat["USERID"])))):
            tmp = dat[dat["USERID"] == id].reset_index()
            if self.msg_type == "all":
                A[i, :] = (tmp["week_category"].values != "None") + 0
            else:
                A[i, :] = (tmp["week_category"].values == self.msg_type) + 0

            R[i, :] = tmp[self.reward_name].values

            M[i, :, :] = tmp[self.mediator_names].to_numpy()
        A = A.astype(int)

        return M, A, R


class IHS2020DatasetMI:
    def __init__(self, specialty, msg_type, reward_name, mediator_names) -> None:
        dir_path = "../data/2020_daily/*.*"
        self.data = {}
        for idx, file in enumerate(glob.glob(dir_path)):
            all_data = pd.read_csv(file, na_filter=False)
            self.data[idx] = all_data
        assert msg_type in ["steps", "sleep", "mood", "all"]
        self.msg_type = msg_type
        assert reward_name in [
            "STEP_COUNT",
            "SLEEP_COUNT",
            "RHR",
            "MOOD",
        ]
        self.reward_name = reward_name
        assert set(mediator_names).issubset(
            set(["STEP_COUNT", "SLEEP_COUNT", "RHR", "MOOD"])
        )
        self.mediator_names = mediator_names
        self.num_mediators = len(self.mediator_names)
        self.N = len(set(self.data[0]["STUDY_PRTCPT_ID"]))
        self.T = len(set(self.data[0]["Day"]))
        self.num_imputations = len(self.data)

    def load_data(self, index):
        dat = (
            self.data[index]
            .sort_values(by=["STUDY_PRTCPT_ID", "Day"])
            .reset_index()
            .copy(deep=True)
        )

        MDIM = len(self.mediator_names)
        M = np.zeros([self.N, self.T, MDIM], dtype=float)
        R = np.zeros([self.N, self.T], dtype=float)
        A = np.zeros([self.N, self.T], dtype=float)
        for id, i in zip(
            sorted(set(dat["STUDY_PRTCPT_ID"])), range(len(set(dat["STUDY_PRTCPT_ID"])))
        ):
            tmp = dat[dat["STUDY_PRTCPT_ID"] == id].reset_index()
            if self.msg_type == "all":
                A[i, :] = (tmp["Notification_type"].values != "nomsg") + 0
            else:
                A[i, :] = (tmp["Notification_type"].values == self.msg_type) + 0

            R[i, :] = tmp[self.reward_name].values

            M[i, :, :] = tmp[self.mediator_names].to_numpy()
        A = A.astype(int)

        return M, A, R
