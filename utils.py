import numpy as np
import pandas as pd


RAW_MATCHES_FILE = 'raw_matches.csv'
RAW_WINNERS_FILE = 'raw_winners.csv'
TEAM_RENAMES_FILE = 'team_renames.csv'


def apply_renames(column):
    """Apply team renames to a team column from a dataframe."""
    with open(TEAM_RENAMES_FILE) as renames_file:
        renames = dict(l.strip().split(',')
                       for l in renames_file.readlines()
                       if l.strip())

        def renamer(team):
            return renames.get(team, team)

    return column.map(renamer)


def get_matches(with_team_stats=False, duplicate_with_reversed=False):
    """Create a dataframe with matches info."""
    matches = pd.read_csv(RAW_MATCHES_FILE)
    for column in ('team1', 'team2'):
        matches[column] = apply_renames(matches[column])

    if duplicate_with_reversed:
        id_offset = len(matches)

        matches2 = matches.copy()
        matches2.rename(columns={'team1': 'team2',
                                 'team2': 'team1',
                                 'score1': 'score2',
                                 'score2': 'score1'},
                        inplace=True)
        matches2.index = matches2.index.map(lambda x: x + id_offset)

        matches = pd.concat((matches, matches2))

    def winner_from_score_diff(x):
        if x > 0:
            return 0
        else:
            return 1

    matches['score_diff'] = matches.score1 - matches.score2
    matches = matches[matches.score_diff != 0]  # remove ties
    matches['winner'] = matches.score_diff.map(winner_from_score_diff)

    if with_team_stats:
        stats = get_team_stats()

        matches = matches.join(stats, on='team1')\
                         .join(stats, on='team2', rsuffix='_2')

    return matches


def get_winners():
    """Create a dataframe with podium positions info."""
    winners = pd.read_csv(RAW_WINNERS_FILE)
    winners.team = apply_renames(winners.team)

    return winners


def get_team_stats():
    """Create a dataframe with useful stats for each team."""
    winners = get_winners()
    matches = get_matches()

    teams = set(matches.team1.unique()).union(matches.team2.unique())

    stats = pd.DataFrame(list(teams), columns=['team'])

    stats = stats.set_index('team')

    for team in teams:
        team_matches = matches[(matches.team1 == team) |
                               (matches.team2 == team)]
        stats.loc[team, 'matches_played'] = len(team_matches)

        # wins where the team was on the left side (team1)
        wins1 = team_matches[(team_matches.team1 == team) &
                             (team_matches.score1 > team_matches.score2)]
        # wins where the team was on the right side (team2)
        wins2 = team_matches[(team_matches.team2 == team) &
                             (team_matches.score2 > team_matches.score1)]

        stats.loc[team, 'matches_won'] = len(wins1) + len(wins2)

        stats.loc[team, 'years_played'] = len(team_matches.year.unique())

        team_podiums = winners[winners.team == team]
        to_score = lambda position: 2 ** (5 - position)  # better position -> more score, exponential
        stats.loc[team, 'podium_score'] = team_podiums.position.map(to_score).sum()

        stats.loc[team, 'cups_won'] = len(team_podiums[team_podiums.position == 1])

    stats['matches_won_percent'] = stats.matches_won / stats.matches_played * 100
    stats['podium_score_yearly'] = stats.podium_score / stats.years_played
    stats['cups_won_yearly'] = stats.cups_won / stats.years_played

    return stats


def build_inputs_for_match(year, team1, team2, input_features):
    """Build the inputs for a single hypothetical match, real or not."""
    inputs = []

    team_stats = get_team_stats()
    for feature in input_features:
        from_team_2 = '_2' in feature
        feature = feature.replace('_2', '')

        if feature in team_stats.columns.values:
            team = team2 if from_team_2 else team1
            value = team_stats.loc[team, feature]
        elif feature == 'year':
            value = year
        else:
            raise ValueError("Don't know where to get feature: " + feature)

        inputs.append(value)

    return np.array([inputs, ])
