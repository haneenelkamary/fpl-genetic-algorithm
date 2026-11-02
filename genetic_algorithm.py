import random
from pathlib import Path

import pandas as pd


# ==========================================
# 1. GENETIC ALGORITHM SETTINGS
# ==========================================

BUDGET = 100.0
POPULATION_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.15
ELITE_SIZE = 10
TOURNAMENT_SIZE = 5

MAX_PLAYERS_PER_TEAM = 3

SQUAD_REQUIREMENTS = {
    "Goalkeeper": 2,
    "Defender": 5,
    "Midfielder": 5,
    "Forward": 3,
}

DATA_FILE = Path("data/players.csv")
OUTPUT_FILE = Path("data/best_squad.csv")


# ==========================================
# 2. LOAD AND PREPARE DATA
# ==========================================

def load_players():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Players data not found: {DATA_FILE}\n"
            "Make sure data/players.csv exists."
        )

    players = pd.read_csv(DATA_FILE)

    required_columns = [
        "player_name",
        "team",
        "position",
        "price",
        "expected_points_next_round",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in players.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Convert numeric columns safely.
    numeric_columns = [
        "price",
        "expected_points_next_round",
        "chance_of_playing",
        "points_per_game",
    ]

    for column in numeric_columns:
        if column not in players.columns:
            players[column] = 0 if column == "points_per_game" else float("nan")

        players[column] = pd.to_numeric(
            players[column],
            errors="coerce",
        )

    # Keep players with valid positions, prices and projections.
    players = players[
        players["position"].isin(SQUAD_REQUIREMENTS)
        & players["price"].notna()
        & players["expected_points_next_round"].notna()
        & (players["price"] > 0)
    ].copy()

    if players.empty:
        raise ValueError(
            "No usable players were found in data/players.csv."
        )

    # Record whether availability information was provided.
    players["availability_known"] = (
        players["chance_of_playing"].notna()
    )

    # Missing availability percentages are treated as 100%.
    players["chance_of_playing"] = (
        players["chance_of_playing"]
        .fillna(100)
        .clip(0, 100)
    )

    # Prevent negative projected points.
    players["expected_points_next_round"] = (
        players["expected_points_next_round"].clip(lower=0)
    )

    # Missing historical points per game become zero.
    players["points_per_game"] = (
        players["points_per_game"].fillna(0).clip(lower=0)
    )

    # Adjust projections according to the chance of playing.
    players["adjusted_points"] = (
        players["expected_points_next_round"]
        * players["chance_of_playing"]
        / 100
    )

    # Reset indexes so player IDs used by the algorithm are consistent.
    players = players.reset_index(drop=True)

    # Make sure there are enough players for every position.
    for position, required_count in SQUAD_REQUIREMENTS.items():
        available_count = (
            players["position"] == position
        ).sum()

        if available_count < required_count:
            raise ValueError(
                f"Not enough players for {position}: "
                f"need {required_count}, found {available_count}."
            )

    return players


# ==========================================
# 3. SQUAD VALIDATION
# ==========================================

def is_valid_squad(squad, players):
    if len(squad) != 15:
        return False

    if len(set(squad)) != 15:
        return False

    if any(index < 0 or index >= len(players) for index in squad):
        return False

    selected = players.iloc[list(squad)]

    # Budget limit.
    if selected["price"].sum() > BUDGET:
        return False

    # Required number of players in each position.
    for position, required_count in SQUAD_REQUIREMENTS.items():
        actual_count = (
            selected["position"] == position
        ).sum()

        if actual_count != required_count:
            return False

    # Maximum of three players from one club.
    if (
        selected["team"].value_counts() > MAX_PLAYERS_PER_TEAM
    ).any():
        return False

    return True


# ==========================================
# 4. SELECT THE CAPTAIN
# ==========================================

def select_captain(candidates):
    """
    Choose the captain using these criteria, in order:

    1. Highest adjusted expected points.
    2. Highest chance of playing.
    3. Highest points per game.
    """

    if candidates.empty:
        return None

    # Prefer players whose availability information is known.
    eligible = candidates[
        candidates["availability_known"]
    ].copy()

    # If no player has an explicit availability percentage,
    # use the available players instead.
    if eligible.empty:
        eligible = candidates.copy()

    # Sort all tie-break criteria together.
    eligible = eligible.sort_values(
        by=[
            "adjusted_points",
            "chance_of_playing",
            "points_per_game",
        ],
        ascending=[False, False, False],
        kind="stable",
    )

    return eligible.index[0]


# ==========================================
# 5. FIND THE BEST STARTING XI
# ==========================================

def get_best_lineup(squad, players):
    """
    Find the best legal starting XI from a 15-player squad.

    The score includes:
    - Expected points from the starting XI.
    - The captain's additional expected points.
    """

    selected = players.loc[list(squad)]

    goalkeepers = selected[
        selected["position"] == "Goalkeeper"
    ].sort_values(
        "adjusted_points",
        ascending=False,
    )

    defenders = selected[
        selected["position"] == "Defender"
    ].sort_values(
        "adjusted_points",
        ascending=False,
    )

    midfielders = selected[
        selected["position"] == "Midfielder"
    ].sort_values(
        "adjusted_points",
        ascending=False,
    )

    forwards = selected[
        selected["position"] == "Forward"
    ].sort_values(
        "adjusted_points",
        ascending=False,
    )

    best_lineup = None
    best_captain = None
    best_score = float("-inf")

    # Try all legal outfield formations.
    for defender_count in range(3, 6):
        for midfielder_count in range(2, 6):

            forward_count = (
                10 - defender_count - midfielder_count
            )

            if not 1 <= forward_count <= 3:
                continue

            # Make sure the required players exist.
            if (
                len(goalkeepers) < 1
                or len(defenders) < defender_count
                or len(midfielders) < midfielder_count
                or len(forwards) < forward_count
            ):
                continue

            lineup = [
                goalkeepers.index[0],
                *defenders.index[:defender_count],
                *midfielders.index[:midfielder_count],
                *forwards.index[:forward_count],
            ]

            starting_players = players.loc[lineup]

            # Choose the captain using the tie-break criteria.
            captain = select_captain(starting_players)

            if captain is None:
                continue

            # Expected points from the starting XI.
            starting_points = (
                starting_players["adjusted_points"].sum()
            )

            # Captain receives an additional set of expected points.
            score = (
                starting_points
                + players.at[captain, "adjusted_points"]
            )

            if score > best_score:
                best_score = float(score)
                best_lineup = lineup
                best_captain = captain

    if best_lineup is None:
        raise RuntimeError(
            "Could not find a legal starting XI. "
            "Check the available players and their positions."
        )

    return best_lineup, best_captain, best_score


# ==========================================
# 6. FITNESS FUNCTION
# ==========================================

def fitness(squad, players):
    """
    Fitness is the expected score of the best starting XI,
    including the captain's additional points.
    """

    if not is_valid_squad(squad, players):
        return -1.0

    _, _, score = get_best_lineup(squad, players)

    return score


# ==========================================
# 7. RANDOM VALID SQUAD GENERATION
# ==========================================

def create_random_squad(players, max_attempts=5000):
    position_candidates = {
        position: players.index[
            players["position"] == position
        ].tolist()
        for position in SQUAD_REQUIREMENTS
    }

    for _ in range(max_attempts):
        chosen = []

        for position, count in SQUAD_REQUIREMENTS.items():
            candidates = position_candidates[position]

            chosen.extend(
                random.sample(candidates, count)
            )

        if is_valid_squad(chosen, players):
            return chosen

    raise RuntimeError(
        "Could not generate a valid squad. "
        "Check player prices, positions, budget and club limits."
    )


def create_initial_population(players):
    population = []

    while len(population) < POPULATION_SIZE:
        population.append(
            create_random_squad(players)
        )

    return population


# ==========================================
# 8. SELECTION
# ==========================================

def select_parent(population, players):
    candidates = random.sample(
        population,
        min(TOURNAMENT_SIZE, len(population)),
    )

    return max(
        candidates,
        key=lambda squad: fitness(squad, players),
    )


# ==========================================
# 9. CROSSOVER
# ==========================================

def crossover(parent1, parent2, players):
    """
    Combine two parent squads while preserving
    the required number of players in each position.
    """

    for _ in range(200):
        child = []

        for position, required_count in SQUAD_REQUIREMENTS.items():

            parent1_players = [
                index for index in parent1
                if players.at[index, "position"] == position
            ]

            parent2_players = [
                index for index in parent2
                if players.at[index, "position"] == position
            ]

            combined = list(
                dict.fromkeys(
                    parent1_players + parent2_players
                )
            )

            random.shuffle(combined)

            # Add more players if there are not enough unique candidates.
            if len(combined) < required_count:
                candidates = players.index[
                    players["position"] == position
                ].tolist()

                random.shuffle(candidates)

                for candidate in candidates:
                    if candidate not in combined:
                        combined.append(candidate)

                    if len(combined) >= required_count:
                        break

            child.extend(
                combined[:required_count]
            )

        if is_valid_squad(child, players):
            return child

    # Fall back to a randomly generated valid squad.
    return create_random_squad(players)


# ==========================================
# 10. MUTATION
# ==========================================

def mutate(squad, players):
    """
    Replace one player with another player
    in the same position, if the new squad is valid.
    """

    child = squad.copy()

    for _ in range(200):
        position = random.choice(
            list(SQUAD_REQUIREMENTS.keys())
        )

        current_position_players = [
            index for index in child
            if players.at[index, "position"] == position
        ]

        if not current_position_players:
            continue

        old_player = random.choice(
            current_position_players
        )

        candidates = players.index[
            (players["position"] == position)
            & (~players.index.isin(child))
        ].tolist()

        random.shuffle(candidates)

        for new_player in candidates:
            trial = child.copy()

            old_index = trial.index(old_player)
            trial[old_index] = new_player

            if is_valid_squad(trial, players):
                return trial

    return child


# ==========================================
# 11. RUN THE GENETIC ALGORITHM
# ==========================================

def run_genetic_algorithm(players):
    population = create_initial_population(players)

    best_squad = max(
        population,
        key=lambda squad: fitness(squad, players),
    ).copy()

    best_score = fitness(best_squad, players)

    for generation in range(GENERATIONS):

        # Evaluate and sort the current population.
        population.sort(
            key=lambda squad: fitness(squad, players),
            reverse=True,
        )

        current_best_score = fitness(
            population[0],
            players,
        )

        if current_best_score > best_score:
            best_squad = population[0].copy()
            best_score = current_best_score

        # Preserve the best individuals (elitism).
        next_population = [
            squad.copy()
            for squad in population[:ELITE_SIZE]
        ]

        while len(next_population) < POPULATION_SIZE:

            parent1 = select_parent(
                population,
                players,
            )

            parent2 = select_parent(
                population,
                players,
            )

            child = crossover(
                parent1,
                parent2,
                players,
            )

            if random.random() < MUTATION_RATE:
                child = mutate(
                    child,
                    players,
                )

            if is_valid_squad(child, players):
                next_population.append(child)

        population = next_population

        if (generation + 1) % 20 == 0:
            print(
                f"Generation {generation + 1}/{GENERATIONS} | "
                f"Best expected points: {best_score:.2f} "
                "(XI + captain)"
            )

    # Check the final population as well.
    final_candidate = max(
        population,
        key=lambda squad: fitness(squad, players),
    )

    final_score = fitness(
        final_candidate,
        players,
    )

    if final_score > best_score:
        best_squad = final_candidate.copy()

    return best_squad


# ==========================================
# 12. DISPLAY FINAL TEAM
# ==========================================

def display_squad(squad, players):
    selected = players.loc[list(squad)].copy()

    lineup, captain, score = get_best_lineup(
        squad,
        players,
    )

    starting_players = players.loc[lineup].copy()

    starting_players["role"] = "Starter"
    starting_players.loc[captain, "role"] = "Captain"

    # Remaining four players form the bench.
    bench = selected.loc[
        ~selected.index.isin(lineup)
    ].copy()

    bench_order = {
        "Goalkeeper": 0,
        "Defender": 1,
        "Midfielder": 2,
        "Forward": 3,
    }

    bench["bench_order"] = (
        bench["position"].map(bench_order)
    )

    bench = bench.sort_values(
        ["bench_order", "adjusted_points"],
        ascending=[True, False],
    )

    bench["role"] = "Bench"

    # Choose the vice-captain from the remaining starters.
    vice_candidates = starting_players.drop(
        index=captain,
        errors="ignore",
    )

    vice_captain = select_captain(
        vice_candidates
    )

    columns = [
        "player_name",
        "team",
        "position",
        "price",
        "expected_points_next_round",
        "adjusted_points",
        "role",
    ]

    print("\n" + "=" * 75)
    print("BEST FPL SQUAD")
    print("=" * 75)

    print("\nSTARTING XI")

    print(
        starting_players[columns]
        .sort_values(
            ["position", "adjusted_points"],
            ascending=[True, False],
        )
        .to_string(index=False)
    )

    print("\nBENCH")

    print(
        bench[columns].to_string(index=False)
    )

    print("\n" + "-" * 75)

    print(
        f"Captain: {players.at[captain, 'player_name']}"
    )

    if vice_captain is not None:
        print(
            "Vice-captain: "
            f"{players.at[vice_captain, 'player_name']}"
        )
    else:
        print("Vice-captain: No eligible player")

    print(
        "Squad cost: £{:.1f}m".format(
            selected["price"].sum()
        )
    )

    print(
        "Budget remaining: £{:.1f}m".format(
            BUDGET - selected["price"].sum()
        )
    )

    print(
        "Starting XI expected points "
        "(before captain bonus): {:.2f}".format(
            starting_players["adjusted_points"].sum()
        )
    )

    print(
        "Starting XI expected points "
        "(including captain bonus): {:.2f}".format(
            score
        )
    )

    # Show the captain tie-break statistics.
    print("\nCAPTAIN DETAILS")

    print(
        f"Player: {players.at[captain, 'player_name']}"
    )

    print(
        "Expected points: "
        f"{players.at[captain, 'expected_points_next_round']:.2f}"
    )

    print(
        "Adjusted points: "
        f"{players.at[captain, 'adjusted_points']:.2f}"
    )

    print(
        "Chance of playing: "
        f"{players.at[captain, 'chance_of_playing']:.1f}%"
    )

    print(
        "Points per game: "
        f"{players.at[captain, 'points_per_game']:.2f}"
    )

    print("=" * 75)

    # Prepare the CSV output.
    result = selected.copy()
    result["role"] = "Bench"

    result.loc[lineup, "role"] = "Starter"
    result.loc[captain, "role"] = "Captain"

    if vice_captain is not None:
        result.loc[vice_captain, "role"] = "Vice-captain"

    return result


# ==========================================
# 13. MAIN
# ==========================================

def main():
    random.seed(42)

    print("Loading FPL players...")

    players = load_players()

    print(f"Available players: {len(players)}")
    print("\nRunning Genetic Algorithm...")
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Generations: {GENERATIONS}")
    print(f"Budget: £{BUDGET:.1f}m")

    best_squad = run_genetic_algorithm(players)

    result = display_squad(
        best_squad,
        players,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nSaved result to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()