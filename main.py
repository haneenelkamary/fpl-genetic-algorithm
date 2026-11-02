import requests
import pandas as pd
from pathlib import Path

BASE_URL = "https://fantasy.premierleague.com/api/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 FPL-Genetic-Algorithm-Student-Project"
}


def get_data(endpoint):
    """Fetch JSON data from the FPL public API."""
    response = requests.get(
        BASE_URL + endpoint,
        headers=HEADERS,
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def main():
    print("Downloading Fantasy Premier League data...")

    # 1. Download players, teams, positions, and gameweeks
    data = get_data("bootstrap-static/")

    players = data["elements"]
    teams = data["teams"]
    positions = data["element_types"]
    events = data["events"]

    # 2. Create lookup dictionaries
    team_names = {
        team["id"]: team["name"]
        for team in teams
    }

    position_names = {
        position["id"]: position["singular_name"]
        for position in positions
    }

    # 3. Identify the next gameweek, if available
    next_event = next(
        (event for event in events if event.get("is_next")),
        None
    )

    if next_event:
        print(
            f"Next gameweek: GW{next_event['id']}"
        )
        print(
            f"Deadline: {next_event.get('deadline_time')}"
        )
    else:
        print("No upcoming gameweek was identified.")

    # 4. Convert raw player data into a table
    rows = []

    for player in players:
        rows.append({
            "player_id": player["id"],
            "player_name": player["web_name"],
            "team": team_names.get(player["team"], "Unknown"),
            "position": position_names.get(
                player["element_type"], "Unknown"
            ),
            "price": player["now_cost"] / 10,
            "status": player.get("status"),
            "chance_of_playing": (
                player.get("chance_of_playing_next_round")
            ),
            "minutes": player.get("minutes", 0),
            "starts": player.get("starts", 0),
            "total_points": player.get("total_points", 0),
            "points_per_game": player.get("points_per_game", "0"),
            "form": player.get("form", "0"),
            "goals_scored": player.get("goals_scored", 0),
            "assists": player.get("assists", 0),
            "expected_goals": player.get("expected_goals", "0"),
            "expected_assists": player.get(
                "expected_assists", "0"
            ),
            "expected_points_next_round": player.get("ep_next"),
            "selected_by_percent": player.get(
                "selected_by_percent", "0"
            ),
        })

    players_df = pd.DataFrame(rows)

    # 5. Convert numeric columns
    numeric_columns = [
        "price",
        "minutes",
        "starts",
        "total_points",
        "points_per_game",
        "form",
        "goals_scored",
        "assists",
        "expected_goals",
        "expected_assists",
        "expected_points_next_round",
        "selected_by_percent",
    ]

    for column in numeric_columns:
        players_df[column] = pd.to_numeric(
            players_df[column],
            errors="coerce"
        )

    # 6. Download fixtures
    fixtures = get_data("fixtures/")

    fixtures_df = pd.DataFrame(fixtures)

    if not fixtures_df.empty:
        fixtures_df["home_team"] = fixtures_df[
            "team_h"
        ].map(team_names)

        fixtures_df["away_team"] = fixtures_df[
            "team_a"
        ].map(team_names)

    # 7. Save datasets
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    players_df.to_csv(
        output_dir / "players.csv",
        index=False
    )

    fixtures_df.to_csv(
        output_dir / "fixtures.csv",
        index=False
    )

    # 8. Display a preview
    print(f"\nPlayers downloaded: {len(players_df)}")
    print(f"Fixtures downloaded: {len(fixtures_df)}")

    print("\nFirst 10 players:")
    print(
        players_df[
            [
                "player_name",
                "team",
                "position",
                "price",
                "total_points",
                "form",
                "expected_points_next_round",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nFiles saved successfully:")
    print("data/players.csv")
    print("data/fixtures.csv")


if __name__ == "__main__":
    main()