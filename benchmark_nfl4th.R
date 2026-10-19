# External benchmark: nfl4th (Baldwin; github.com/nflverse/nfl4th) recommendations for every
# 2025 fourth down in the same nflverse file the Python pipeline uses.
#   Rscript benchmark_nfl4th.R      # needs data/pbp_2025.csv.gz (python fourth_down.py build --source nflverse)
# Output: data/nfl4th_2025_recommendations.csv (shipped in the repo so the Python build needs no R).
# nfl4th's go-for-it model was trained on 2014-2019 seasons, so 2025 is out of sample for it too.
suppressPackageStartupMessages({library(dplyr); library(nfl4th)})
# nfl4th looks up game info (roof, surface) through nflreadr::load_schedules(). If that download is
# blocked, point it at a local copy of nflverse/nfldata's games.rds (set NFLDATA_GAMES=path).
local_games <- Sys.getenv("NFLDATA_GAMES")
if (nzchar(local_games)) {
  assignInNamespace("load_schedules", function(...) {
    nflreadr:::make_nflverse_data(readRDS(local_games), nflverse_type = "games and schedules")
  }, ns = "nflreadr")
}
pbp <- data.table::fread("data/pbp_2025.csv.gz", showProgress = FALSE) |> as_tibble()
fd <- pbp |> filter(down == 4, play_type %in% c("run", "pass", "punt", "field_goal"), qtr <= 4)
cat("fourth downs:", nrow(fd), "\n")
out <- nfl4th::add_4th_probs(fd)
res <- out |>
  transmute(game_id, play_id,
            nfl4th_go_wp = go_wp, nfl4th_fg_wp = fg_wp, nfl4th_punt_wp = punt_wp) |>
  mutate(nfl4th_rec = case_when(
    is.na(nfl4th_go_wp) ~ NA_character_,
    nfl4th_go_wp >= pmax(coalesce(nfl4th_fg_wp, -1), coalesce(nfl4th_punt_wp, -1)) ~ "go",
    coalesce(nfl4th_fg_wp, -1) >= coalesce(nfl4th_punt_wp, -1) ~ "field_goal",
    TRUE ~ "punt"))
data.table::fwrite(res, "data/nfl4th_2025_recommendations.csv")
cat("wrote", nrow(res), "rows;", sum(!is.na(res$nfl4th_rec)), "with a recommendation\n")
cat("nfl4th", as.character(packageVersion("nfl4th")), "nflfastR", as.character(packageVersion("nflfastR")),
    "xgboost", as.character(packageVersion("xgboost")), "R", R.version.string, "\n")
print(table(res$nfl4th_rec, useNA = "ifany"))
