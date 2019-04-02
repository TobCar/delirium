"""
@author: Max Berkowitz
"""


def drop_some_subjects(df):
    # We drop subjects without a clear label
    df = df[df.subject_id != "confocal_2"]
    df = df[df.subject_id != "confocal_7"]
    df = df[df.subject_id != "confocal_41"]
    df = df[df.subject_id != "confocal_48"]
    df = df[df.subject_id != "confocal_1"]  # artMAP only has NaNs
    df = df[df.subject_id != "confocal_8"]  # artMAP only has NaNs
    df = df[df.subject_id != "confocal_12"]  # artMAP only has NaNs
    df = df[df.subject_id != "confocal_20"]  # artMAP only has NaNs
    df = df[df.subject_id != "confocal_21"]  # artMAP only has NaNs
    df = df[df.subject_id != "confocal_45"]  # artMAP only has NaNs
    df = df[df.subject_id != "confocal_54"]  # artMAP only has NaNs
    df = df[df.subject_id != "confocal_49"]  # So many NaNs every observation has at least one NaN
    df = df[df.subject_id != "confocal_46"]  # So many NaNs every observation has at least one NaN
    df = df[df.subject_id != "confocal_30"]  # Was unable to screen, cannot be certain in the label
    df = df[df.subject_id != "confocal_56"]  # Was unable to screen, cannot be certain in the label. Was never comatose
    return df
