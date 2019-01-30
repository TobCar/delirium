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
    return df
