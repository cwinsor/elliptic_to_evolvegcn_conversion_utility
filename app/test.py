
import pandas as pd


def main():

    df = pd.DataFrame([
        [0, 0, 0, 0],
        [0, 0, 6, 6],
        [0, 0, 7, 7],
        [0, 0, 0, 0]])

    get_timestep_check = lambda x: x.iloc[0] != x.iloc[1]

    print(df[[0, 2]])

    df['not_equal'] = df[[1, 2]].apply(get_timestep_check, axis=1)
    print(df)

    print("count total: {}".format(df.shape[0]))
    select = df['not_equal']
    print("count of mismatches = {}".format(df[select].shape[0]))
    print(df[select])
    print(df['not_equal'].any())


if __name__ == "__main__":

    main()
