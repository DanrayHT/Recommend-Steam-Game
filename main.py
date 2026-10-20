import pandas as pd

if __name__ == "__main__":
    # create_steam_info(8)
    df_game = pd.read_csv("data/game_info.csv")
    print(df_game['review'].unique())