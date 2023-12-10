import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

leaderboard_path = "worlds2022_standing.csv"

tournament_urls = {
    "LEC": "https://gol.gg/tournament/tournament-ranking/LEC%20Summer%202022/",
    "LCS": "https://gol.gg/tournament/tournament-ranking/LCS%20Summer%202022/",
    "LPL": "https://gol.gg/tournament/tournament-ranking/LPL%20Summer%202022/",
    "LCK": "https://gol.gg/tournament/tournament-ranking/LCK%20Summer%202022/"
}

# worlds = "https://gol.gg/tournament/tournament-ranking/World%20Championship%202022/"

# GPM = Gold per Minute

def scrape_league_data(url):
    page = requests.get(url)
    html_page = BeautifulSoup(page.content, "html.parser")

    return html_page

def parse_teams_names(html_page):
    pattern = re.compile(r'.* stats in .* 2022')
    league_table = html_page.find_all(title=pattern)

    teams = []

    for team in league_table:
        team_name = team.text
        teams.append(team_name)

    return teams

def parse_teams_stats(html_page):
    team_rows = html_page.find_all('tr')[1:]
    teams_stats = []

    for row in team_rows:
        data_cells = row.find_all('td', class_='text-center')
        for data_cell in data_cells:
            stats = data_cell.text.strip()
            teams_stats.append(stats)

    return teams_stats

def combine_data(teams, teams_stats):
    result = []

    for i in range(len(teams)):
        name = teams[i]
        start_index = i * 5
        stats_for_name = teams_stats[start_index:start_index+5]
        name_and_stats = [name] + stats_for_name
        result.append(name_and_stats)

    return result

def data_frame_from_csv(path):
    leaderboard = pd.read_csv(path, names = ['team_name', 'winrate', 'placement'])
    
    return leaderboard

def data_frame_from_list(list):
    df = pd.DataFrame(list, columns = ['team_name', 'winrate', 'wins', 'loses', 'game duration', 'GPM'])

    return df


def merge_data_frames(tournament_urls):
    data_frames = []

    for league, url in tournament_urls.items():
        html_page = scrape_league_data(url)
        teams = parse_teams_names(html_page)
        teams_stats = parse_teams_stats(html_page)
        result = combine_data(teams, teams_stats)
        print(result)

        league = data_frame_from_list(result)
        league['standing'] = [(i+1) for i in range(len(league))]
        data_frames.append(league)

    merged_data_frames = pd.concat(data_frames, ignore_index=True)

    return merged_data_frames

def plot_relation_GPM_worldstading(merged_data_frames, leaderboard):
    common_teams = list(set(merged_data_frames['team_name']).intersection(leaderboard['team_name']))

    common_teams_data_merged = merged_data_frames[merged_data_frames['team_name'].isin(common_teams)][['team_name', 'GPM']]
    common_teams_data_leaderboard = leaderboard[leaderboard['team_name'].isin(common_teams)][['team_name', 'placement']]

    result_df = pd.concat([common_teams_data_merged.set_index('team_name'), common_teams_data_leaderboard.set_index('team_name')], axis=1, join='inner').reset_index()
    result_df.columns = ['team_name', 'GPM', 'standing']
    # print(result_df)
    
    result_df['GPM'] = pd.to_numeric(result_df['GPM'], errors='coerce')
    result_df['standing'] = pd.to_numeric(result_df['standing'], errors='coerce')

    x = result_df['GPM']
    y = result_df['standing']

    plt.scatter(x, y, color='black')
    plt.xlabel('Gold per minute')
    plt.ylabel('Final standing at worlds 2022')
    plt.title('Correlation between GPM and worlds standing')

    plt.gca().invert_yaxis()
    plt.show()

def ml_algorithm_data_preparation(merged_data_frames, leaderboard):
    common_teams = list(set(merged_data_frames['team_name']).intersection(leaderboard['team_name']))

    common_teams_merged_data = merged_data_frames[merged_data_frames['team_name'].isin(common_teams)][['team_name', 'winrate', 'wins', 'loses', 'GPM', 'standing']]
    common_teams_leaderboard = leaderboard[leaderboard['team_name'].isin(common_teams)][['team_name', 'placement']]

    data_for_algorithm = pd.concat([common_teams_merged_data.set_index('team_name'), common_teams_leaderboard.set_index('team_name')], axis=1, join='inner').reset_index()

    data_for_algorithm['winrate'] = data_for_algorithm['winrate'].str.strip('%').astype(int)

    data_for_algorithm['GPM'] = pd.to_numeric(data_for_algorithm['GPM'], errors='coerce')
    data_for_algorithm['winrate'] = pd.to_numeric(data_for_algorithm['winrate'], errors='coerce')
    data_for_algorithm['wins'] = pd.to_numeric(data_for_algorithm['wins'], errors='coerce')
    data_for_algorithm['loses'] = pd.to_numeric(data_for_algorithm['loses'], errors='coerce')

    data_for_algorithm = data_for_algorithm.drop('team_name', axis=1)
    data_for_algorithm = pd.concat([pd.Series(1, index=data_for_algorithm.index, name='param0'), data_for_algorithm], axis=1)

    X = data_for_algorithm.drop(columns='placement')
    Y = data_for_algorithm.iloc[:, 6]

    X = X.apply(lambda x: x / np.max(x))

    return X, Y, data_for_algorithm

def gradientDescent(X, Y, theta, alpha, iterations):
    m = len(Y)
    J_history = []

    for i in range(iterations):
        hypothesis = np.dot(X, theta)
        loss = hypothesis - Y
        gradient = np.dot(X.T, loss) / m
        theta -= alpha * gradient

        cost = np.sum(loss ** 2) / (2 * m)
        J_history.append(cost)

        print(J_history)

    return theta


def main():

    leaderboard = data_frame_from_csv(leaderboard_path)
    merged_data_frames = merge_data_frames(tournament_urls)

    print(merged_data_frames)
    print(leaderboard)

    plot_relation_GPM_worldstading(merged_data_frames, leaderboard)

    X, Y, data_for_algorithm = ml_algorithm_data_preparation(merged_data_frames, leaderboard)
    print(data_for_algorithm)

    print(X)
    print(Y)

    theta = np.zeros(X.shape[1])

    theta = gradientDescent(X, Y, theta, alpha=0.01, iterations=50)
    print('Optimized theta: ', theta)

    theta_max = np.max(theta)
    normalized_theta = [value/theta_max for value in theta]
    print('Normalized theta: ', normalized_theta)

    # BLG, JDG, WEIBO, T1, KT, GENG, LNG, NRG
    # regular season stats 
    # theta0, winrate, wins, loses, GPM, standing
    
    result_checking_data = [
    [1, 81, 30, 7, 255, 1],
    [1, 81, 29, 7, 249, 2],
    [1, 64, 25, 14, 113, 5],
    [1, 50, 21, 21, 60, 5],
    [1, 85, 35, 6, 248, 1],
    [1, 82, 32, 7, 275, 2],
    [1, 68, 25, 12, 155, 3],
    [1, 50, 9, 9, -45, 5]
    ]

    numpy_result_checking_data = np.array(result_checking_data)

    multiplied_result = numpy_result_checking_data[:, 1:] * theta[1:]

    print("Multiplied result:")
    print(multiplied_result)

    sum_per_row = np.sum(multiplied_result, axis=1)

    print("Sum for each row:")
    print(sum_per_row)

    # RESULT 3, 4, 6, 7, 2, 1, 5, 8
    # BLG, JDG, WEIBO, T1, KT, GENG, LNG, NRG

    """
    GENG
    KT
    BLG
    JDG

    LNG
    WEIBO
    T1
    NRG
    """
main()