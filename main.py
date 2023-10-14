import requests
from bs4 import BeautifulSoup
import re

url_lec = "https://gol.gg/tournament/tournament-ranking/LEC%20Summer%202022/"

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

def main():
    html_page = scrape_league_data(url_lec)
    teams = parse_teams_names(html_page)
    teams_stats = parse_teams_stats(html_page)
    result = combine_data(teams, teams_stats)

    print(result)

main()

    

