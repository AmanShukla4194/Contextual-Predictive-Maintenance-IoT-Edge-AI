import pandas as pd


def failure_distribution(failures):
    return failures["failure"].value_counts()


def machine_failure_counts(failures):
    return failures.groupby("machineID")["failure"].count().sort_values(ascending=False)


def telemetry_stats(telemetry, sensor_cols):
    return telemetry[sensor_cols].describe()


def error_frequency(errors):
    return errors["errorID"].value_counts()


def machines_by_model(machines):
    return machines["model"].value_counts()