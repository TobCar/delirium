"""
@author: Max Berkowitz
"""


def drop_some_subjects(df):
    # We drop subjects without a clear label
    df = df[df.subject_id != "confocal_2"]
    df = df[df.subject_id != "confocal_7"]
    df = df[df.subject_id != "confocal_41"]
    df = df[df.subject_id != "confocal_48"]
    df = df[df.subject_id != "confocal_20"]
    return df
