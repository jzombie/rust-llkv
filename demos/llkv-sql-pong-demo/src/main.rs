// TODO: If running in development mode, include warning about unoptimized performance.

use std::cell::RefCell;
use std::io::{Write, stdout};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use std::{cmp, thread};

use arrow::array::{Array, Int64Array};
use arrow::record_batch::RecordBatch;
use crossterm::QueueableCommand;
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{Event, KeyCode, KeyEvent, poll, read};
use crossterm::execute;
use crossterm::queue;
use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use indoc::indoc;
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;
use rustc_hash::FxHashMap;

#[derive(thiserror::Error, Debug)]
enum Error {
    #[error("query returned no batches")]
    QueryReturnedNoBatches,
    #[error("expected exactly one row")]
    UnexpectedRowCount,
    #[error(transparent)]
    Llkv(#[from] llkv_result::error::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

type Result<T> = std::result::Result<T, Error>;

const FULL_BLOCK: char = '\u{2588}';
const HALF_BLOCK: char = '\u{2580}';

const STATE_ROW_ID: i64 = 1;
const APP_TITLE: &str = env!("CARGO_PKG_NAME");

const SETUP_SQL: &str = indoc! {"
    CREATE TABLE params (
      w INT,
      h INT,
      paddle_h INT,
      paddle_speed INT
    );
    INSERT INTO params (w, h, paddle_h, paddle_speed) VALUES (80, 25, 5, 2);
    CREATE TABLE state (
        id INT PRIMARY KEY,
      tick INT,
      ax INT,
      bx INT,
      ball_x INT,
      ball_y INT,
      vx INT,
      vy INT,
      score_a INT,
      score_b INT
    );
    CREATE TABLE digits (
      digit INT,
      row INT,
      pattern TEXT
    );
    INSERT INTO digits (digit, row, pattern) VALUES
      (0, 0, 'FFF'), (0, 1, 'F F'), (0, 2, 'F F'), (0, 3, 'F F'), (0, 4, 'FFF'),
      (1, 0, ' F '), (1, 1, 'FF '), (1, 2, ' F '), (1, 3, ' F '), (1, 4, 'FFF'),
      (2, 0, 'FFF'), (2, 1, '  F'), (2, 2, 'FFF'), (2, 3, 'F  '), (2, 4, 'FFF'),
      (3, 0, 'FFF'), (3, 1, '  F'), (3, 2, 'FFF'), (3, 3, '  F'), (3, 4, 'FFF'),
      (4, 0, 'F F'), (4, 1, 'F F'), (4, 2, 'FFF'), (4, 3, '  F'), (4, 4, '  F'),
      (5, 0, 'FFF'), (5, 1, 'F  '), (5, 2, 'FFF'), (5, 3, '  F'), (5, 4, 'FFF'),
      (6, 0, 'FFF'), (6, 1, 'F  '), (6, 2, 'FFF'), (6, 3, 'F F'), (6, 4, 'FFF'),
      (7, 0, 'FFF'), (7, 1, '  F'), (7, 2, '  F'), (7, 3, '  F'), (7, 4, '  F'),
      (8, 0, 'FFF'), (8, 1, 'F F'), (8, 2, 'FFF'), (8, 3, 'F F'), (8, 4, 'FFF'),
      (9, 0, 'FFF'), (9, 1, 'F F'), (9, 2, 'FFF'), (9, 3, '  F'), (9, 4, 'FFF');
"};

#[derive(Clone, Copy)]
struct Params {
    w: i64,
    h: i64,
    paddle_h: i64,
}

#[derive(Clone, Copy)]
struct State {
    ax: i64,
    bx: i64,
    ball_x: i64,
    ball_y: i64,
    vx: i64,
    score_a: i64,
    score_b: i64,
}

#[derive(Clone)]
struct HudCell {
    x: u16,
    y: u16,
    text: String,
    color: Color,
    bold: bool,
    persist: bool,
}

struct HudCacheEntry {
    text: String,
    width: usize,
}

thread_local! {
    static HUD_CACHE: RefCell<FxHashMap<(u16, u16), HudCacheEntry>> = RefCell::new(FxHashMap::default());
}

struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let mut out = stdout();
        let _ = execute!(out, Show, LeaveAlternateScreen);
        let _ = disable_raw_mode();
    }
}

#[derive(Default)]
struct PerfStats {
    frame_ms: f64,
    update_ms: f64,
    render_ms: f64,
}

impl PerfStats {
    fn record(&mut self, frame: Duration, update: Duration, render: Duration) {
        let alpha = 0.1;
        let frame_ms = frame.as_secs_f64() * 1000.0;
        let update_ms = update.as_secs_f64() * 1000.0;
        let render_ms = render.as_secs_f64() * 1000.0;

        self.frame_ms = Self::smooth(self.frame_ms, frame_ms, alpha);
        self.update_ms = Self::smooth(self.update_ms, update_ms, alpha);
        self.render_ms = Self::smooth(self.render_ms, render_ms, alpha);
    }

    fn smooth(current: f64, sample: f64, alpha: f64) -> f64 {
        if current == 0.0 {
            sample
        } else {
            current * (1.0 - alpha) + sample * alpha
        }
    }

    fn fps(&self) -> f64 {
        if self.frame_ms > 0.0 {
            1000.0 / self.frame_ms
        } else {
            0.0
        }
    }

    fn frame_ms(&self) -> f64 {
        self.frame_ms
    }

    fn update_ms(&self) -> f64 {
        self.update_ms
    }

    fn render_ms(&self) -> f64 {
        self.render_ms
    }
}

fn tick_sql_template() -> &'static str {
    static TEMPLATE: OnceLock<String> = OnceLock::new();
    TEMPLATE.get_or_init(build_tick_sql_template)
}

fn build_tick_sql_template() -> String {
    indoc! {"
        UPDATE state SET
    tick = tick + 1,
    
    -- Left paddle: aggressive play with edge targeting when ball approaches (vx < 0)
    ax = CASE
        WHEN vx < 0 THEN
            -- Aim for top or bottom edge based on random value to create sharp angles
            -- Also clamp to keep paddle on screen
            CASE
                WHEN ball_y - CASE WHEN RANDOM() < 0.5 THEN 0 ELSE 4 END - ax > 2 THEN 
                    CASE WHEN ax + 2 > (SELECT h FROM params LIMIT 1) - (SELECT paddle_h FROM params LIMIT 1) 
                        THEN (SELECT h FROM params LIMIT 1) - (SELECT paddle_h FROM params LIMIT 1)
                        ELSE ax + 2 
                    END
                WHEN ball_y - CASE WHEN RANDOM() < 0.5 THEN 0 ELSE 4 END - ax < -2 THEN 
                    CASE WHEN ax - 2 < 0 THEN 0 ELSE ax - 2 END
                ELSE ax
            END
        ELSE ax
    END,
    
    -- Right paddle: predictive defense with occasional errors for balance (vx > 0)
    bx = CASE
        WHEN vx > 0 THEN
            -- Predict where ball will be, but add random error to make it beatable
            -- 30% of the time, use delayed reaction (just follow current position)
            -- 70% of the time, use prediction with potential overshoot
            CASE
                WHEN RANDOM() < 0.3 THEN
                    -- Delayed reaction: just track current ball position
                    CASE
                        WHEN ball_y - 2 - bx > 2 THEN 
                            CASE WHEN bx + 2 > (SELECT h FROM params LIMIT 1) - (SELECT paddle_h FROM params LIMIT 1) 
                                THEN (SELECT h FROM params LIMIT 1) - (SELECT paddle_h FROM params LIMIT 1)
                                ELSE bx + 2 
                            END
                        WHEN ball_y - 2 - bx < -2 THEN 
                            CASE WHEN bx - 2 < 0 THEN 0 ELSE bx - 2 END
                        ELSE bx
                    END
                ELSE
                    -- Predictive mode with occasional overshoot
                    CASE
                        WHEN CASE
                            WHEN ball_y + (vy * (((SELECT w FROM params LIMIT 1) - 2) - ball_x)) < 1 THEN 1
                            WHEN ball_y + (vy * (((SELECT w FROM params LIMIT 1) - 2) - ball_x)) > (SELECT h FROM params LIMIT 1) - 2 
                                THEN (SELECT h FROM params LIMIT 1) - 2
                            ELSE ball_y + (vy * (((SELECT w FROM params LIMIT 1) - 2) - ball_x))
                        END - 2 - bx > 2 THEN 
                            CASE WHEN bx + 2 > (SELECT h FROM params LIMIT 1) - (SELECT paddle_h FROM params LIMIT 1) 
                                THEN (SELECT h FROM params LIMIT 1) - (SELECT paddle_h FROM params LIMIT 1)
                                ELSE bx + 2 
                            END
                        WHEN CASE
                            WHEN ball_y + (vy * (((SELECT w FROM params LIMIT 1) - 2) - ball_x)) < 1 THEN 1
                            WHEN ball_y + (vy * (((SELECT w FROM params LIMIT 1) - 2) - ball_x)) > (SELECT h FROM params LIMIT 1) - 2 
                                THEN (SELECT h FROM params LIMIT 1) - 2
                            ELSE ball_y + (vy * (((SELECT w FROM params LIMIT 1) - 2) - ball_x))
                        END - 2 - bx < -2 THEN 
                            CASE WHEN bx - 2 < 0 THEN 0 ELSE bx - 2 END
                        ELSE bx
                    END
            END
        ELSE bx
    END,
    
    -- Ball X position with paddle collision
    ball_x = CASE
        -- Left paddle hit - bounce back
        WHEN ball_x + vx <= 1 AND vx < 0 
            AND ball_y >= ax 
            AND ball_y <= ax + (SELECT paddle_h FROM params LIMIT 1) - 1 
            THEN 2
        -- Right paddle hit - bounce back
        WHEN ball_x + vx >= (SELECT w FROM params LIMIT 1) - 2 AND vx > 0 
            AND ball_y >= bx 
            AND ball_y <= bx + (SELECT paddle_h FROM params LIMIT 1) - 1 
            THEN (SELECT w FROM params LIMIT 1) - 3
        -- Right scored (left missed) - only if paddle didn't catch
        WHEN ball_x + vx < 1 
            AND NOT (ball_y >= ax AND ball_y <= ax + (SELECT paddle_h FROM params LIMIT 1) - 1)
            THEN (SELECT w FROM params LIMIT 1) / 2 - 1
        -- Left scored (right missed) - only if paddle didn't catch
        WHEN ball_x + vx > (SELECT w FROM params LIMIT 1) - 2 
            AND NOT (ball_y >= bx AND ball_y <= bx + (SELECT paddle_h FROM params LIMIT 1) - 1)
            THEN (SELECT w FROM params LIMIT 1) / 2 + 1
        ELSE ball_x + vx
    END,
    
    ball_y = CASE
        -- Left paddle hit - continue with velocity
        WHEN ball_x + vx <= 1 AND vx < 0 
            AND ball_y >= ax 
            AND ball_y <= ax + (SELECT paddle_h FROM params LIMIT 1) - 1 
            THEN 
                CASE
                    WHEN ball_y + vy < 1 THEN 1
                    WHEN ball_y + vy > (SELECT h FROM params LIMIT 1) - 2 THEN (SELECT h FROM params LIMIT 1) - 2
                    ELSE ball_y + vy
                END
        -- Right paddle hit - continue with velocity
        WHEN ball_x + vx >= (SELECT w FROM params LIMIT 1) - 2 AND vx > 0 
            AND ball_y >= bx 
            AND ball_y <= bx + (SELECT paddle_h FROM params LIMIT 1) - 1 
            THEN 
                CASE
                    WHEN ball_y + vy < 1 THEN 1
                    WHEN ball_y + vy > (SELECT h FROM params LIMIT 1) - 2 THEN (SELECT h FROM params LIMIT 1) - 2
                    ELSE ball_y + vy
                END
        -- Right scored - reset (only if paddle didn't catch)
        WHEN ball_x + vx < 1 
            AND NOT (ball_y >= ax AND ball_y <= ax + (SELECT paddle_h FROM params LIMIT 1) - 1)
            THEN
                CASE
                    WHEN ((SELECT h FROM params LIMIT 1) / 2) + (CAST(FLOOR(RANDOM() * 7) AS INTEGER) - 3) < 1 THEN 1
                    WHEN ((SELECT h FROM params LIMIT 1) / 2) + (CAST(FLOOR(RANDOM() * 7) AS INTEGER) - 3) > (SELECT h FROM params LIMIT 1) - 2 
                        THEN (SELECT h FROM params LIMIT 1) - 2
                    ELSE ((SELECT h FROM params LIMIT 1) / 2) + (CAST(FLOOR(RANDOM() * 7) AS INTEGER) - 3)
                END
        -- Left scored - reset (only if paddle didn't catch)
        WHEN ball_x + vx > (SELECT w FROM params LIMIT 1) - 2 
            AND NOT (ball_y >= bx AND ball_y <= bx + (SELECT paddle_h FROM params LIMIT 1) - 1)
            THEN 
                CASE
                    WHEN ((SELECT h FROM params LIMIT 1) / 2) + (CAST(FLOOR(RANDOM() * 7) AS INTEGER) - 3) < 1 THEN 1
                    WHEN ((SELECT h FROM params LIMIT 1) / 2) + (CAST(FLOOR(RANDOM() * 7) AS INTEGER) - 3) > (SELECT h FROM params LIMIT 1) - 2 
                        THEN (SELECT h FROM params LIMIT 1) - 2
                    ELSE ((SELECT h FROM params LIMIT 1) / 2) + (CAST(FLOOR(RANDOM() * 7) AS INTEGER) - 3)
                END
        -- Normal movement with wall bounce
        ELSE 
            CASE
                WHEN ball_y + vy < 1 THEN 1
                WHEN ball_y + vy > (SELECT h FROM params LIMIT 1) - 2 THEN (SELECT h FROM params LIMIT 1) - 2
                ELSE ball_y + vy
            END
    END,
    
    -- Horizontal velocity
    vx = CASE
        WHEN ball_x + vx <= 1 AND vx < 0 
            AND ball_y >= ax 
            AND ball_y <= ax + (SELECT paddle_h FROM params LIMIT 1) - 1 THEN 1
        WHEN ball_x + vx >= (SELECT w FROM params LIMIT 1) - 2 AND vx > 0 
            AND ball_y >= bx 
            AND ball_y <= bx + (SELECT paddle_h FROM params LIMIT 1) - 1 THEN -1
        WHEN ball_x + vx < 1 THEN 1
        WHEN ball_x + vx > (SELECT w FROM params LIMIT 1) - 2 THEN -1
        ELSE vx
    END,
    
    -- Vertical velocity with paddle angle physics and loop-breaking nudges
    vy = CASE
        WHEN ball_x + vx <= 1 AND vx < 0 
            AND ball_y >= ax 
            AND ball_y <= ax + (SELECT paddle_h FROM params LIMIT 1) - 1 THEN
            CASE
                WHEN ball_y - ax = 0 THEN -2
                WHEN ball_y - ax = 1 THEN -1
                WHEN ball_y - ax = 2 THEN 
                    -- Center hit: add random nudge to break loops (20% chance)
                    CASE WHEN RANDOM() < 0.2 THEN 
                        CASE WHEN RANDOM() < 0.5 THEN -1 ELSE 1 END
                    ELSE 0 END
                WHEN ball_y - ax = 3 THEN 1
                ELSE 2
            END
        WHEN ball_x + vx >= (SELECT w FROM params LIMIT 1) - 2 AND vx > 0 
            AND ball_y >= bx 
            AND ball_y <= bx + (SELECT paddle_h FROM params LIMIT 1) - 1 THEN
            CASE
                WHEN ball_y - bx = 0 THEN -2
                WHEN ball_y - bx = 1 THEN -1
                WHEN ball_y - bx = 2 THEN 
                    -- Center hit: add random nudge to break loops (20% chance)
                    CASE WHEN RANDOM() < 0.2 THEN 
                        CASE WHEN RANDOM() < 0.5 THEN -1 ELSE 1 END
                    ELSE 0 END
                WHEN ball_y - bx = 3 THEN 1
                ELSE 2
            END
        WHEN ball_x + vx < 1 THEN CAST(FLOOR(RANDOM() * 5) AS INTEGER) - 2
        WHEN ball_x + vx > (SELECT w FROM params LIMIT 1) - 2 THEN CAST(FLOOR(RANDOM() * 5) AS INTEGER) - 2
        WHEN (ball_y + vy < 1 OR ball_y + vy > (SELECT h FROM params LIMIT 1) - 2) THEN -vy
        ELSE vy
    END,
    
    score_a = score_a + CASE 
        WHEN ball_x + vx < 1 
            AND NOT (ball_y >= ax AND ball_y <= ax + (SELECT paddle_h FROM params LIMIT 1) - 1)
        THEN 1 ELSE 0 END,
    score_b = score_b + CASE 
        WHEN ball_x + vx > (SELECT w FROM params LIMIT 1) - 2 
            AND NOT (ball_y >= bx AND ball_y <= bx + (SELECT paddle_h FROM params LIMIT 1) - 1)
        THEN 1 ELSE 0 END
    
        WHERE id = 1;
    "}.to_string()
}

fn main() -> Result<()> {
    let pager = Arc::new(MemPager::default());
    let engine = SqlEngine::new(pager);

    engine.execute(SETUP_SQL)?;
    let params = fetch_params(&engine)?;
    insert_initial_state(&engine, &params)?;
    let mut state = fetch_state(&engine)?;

    enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen, Hide)?;
    let _guard = TerminalGuard;

    let mut fps: u32 = 30;
    let mut frame_dt = Duration::from_secs_f64(1.0 / fps as f64);
    let min_fps: u32 = 15;
    let mut max_mode = false;
    let mut sound_enabled = false;
    let mut last_paddle_beep = Instant::now();
    let min_beep_interval = Duration::from_secs_f64(1.0 / 120.0);
    let mut perf = PerfStats::default();

    loop {
        let frame_start = Instant::now();

        while poll(Duration::from_millis(0))? {
            if let Event::Key(KeyEvent {
                code, modifiers, ..
            }) = read()?
            {
                match (code, modifiers) {
                    (KeyCode::Esc, _) => return Ok(()),
                    (KeyCode::Char('s'), _) | (KeyCode::Char('S'), _) => {
                        sound_enabled = !sound_enabled;
                    }
                    (KeyCode::Char('+'), _) => {
                        if fps == 120 {
                            max_mode = true;
                            frame_dt = Duration::from_secs(0);
                        } else if !max_mode {
                            fps = cmp::min(fps * 2, 120);
                            frame_dt = Duration::from_secs_f64(1.0 / fps as f64);
                        }
                    }
                    (KeyCode::Char('-'), _) => {
                        if max_mode {
                            max_mode = false;
                            fps = 120;
                            frame_dt = Duration::from_secs_f64(1.0 / fps as f64);
                        } else if fps / 2 >= min_fps {
                            fps /= 2;
                            frame_dt = Duration::from_secs_f64(1.0 / fps as f64);
                        }
                    }
                    _ => {}
                }
            }
        }

        let update_start = Instant::now();
        let previous_state = state;
        engine.execute(tick_sql_template())?;
        let updated_state = fetch_state(&engine)?;
        let update_time = update_start.elapsed();

        let point_to = if updated_state.score_a > previous_state.score_a {
            Some('A')
        } else if updated_state.score_b > previous_state.score_b {
            Some('B')
        } else {
            None
        };

        let paddle_bounce = point_to.is_none() && updated_state.vx != previous_state.vx;
        state = updated_state;

        let render_start = Instant::now();
        render_frame(
            &engine,
            &params,
            &state,
            fps,
            sound_enabled,
            max_mode,
            &perf,
        )?;
        let render_time = render_start.elapsed();

        if sound_enabled && !max_mode {
            if point_to.is_some() {
                beep();
                thread::sleep(Duration::from_millis(500));
            } else if paddle_bounce {
                let now = Instant::now();
                if now.duration_since(last_paddle_beep) >= min_beep_interval {
                    beep();
                    last_paddle_beep = now;
                }
            }
        }

        let mut frame_time = frame_start.elapsed();

        if !max_mode && frame_time < frame_dt {
            thread::sleep(frame_dt - frame_time);
            frame_time = frame_start.elapsed();
        }

        perf.record(frame_time, update_time, render_time);
    }
}

fn fetch_params(engine: &SqlEngine) -> Result<Params> {
    let batches = engine.sql("SELECT w, h, paddle_h FROM params LIMIT 1;")?;
    let batch = first_batch(&batches)?;
    ensure_single_row(&batch)?;
    Ok(Params {
        w: value_at(&batch, 0),
        h: value_at(&batch, 1),
        paddle_h: value_at(&batch, 2),
    })
}

/// Initialize the game state using SQL with RANDOM() for randomization.
/// This offloads all randomization to the SQL engine instead of using Rust RNG.
fn insert_initial_state(engine: &SqlEngine, _params: &Params) -> Result<()> {
    // Use INSERT...SELECT directly from params table with all randomization in SQL
    engine.execute(&indoc::formatdoc! {"
        INSERT INTO state (id, tick, ax, bx, ball_x, ball_y, vx, vy, score_a, score_b)
        SELECT 
            {STATE_ROW_ID},
            0,
            (h - paddle_h) / 2,
            (h - paddle_h) / 2,
            w / 2,
            CASE
                WHEN (h / 2) + (CAST(FLOOR(RANDOM() * 7) AS INTEGER) - 3) < 1 THEN 1
                WHEN (h / 2) + (CAST(FLOOR(RANDOM() * 7) AS INTEGER) - 3) > h - 2 THEN h - 2
                ELSE (h / 2) + (CAST(FLOOR(RANDOM() * 7) AS INTEGER) - 3)
            END,
            CASE WHEN RANDOM() < 0.5 THEN 1 ELSE -1 END,
            CAST(FLOOR(RANDOM() * 5) AS INTEGER) - 2,
            0,
            0
        FROM params
        LIMIT 1;
    "})?;

    Ok(())
}

fn fetch_state(engine: &SqlEngine) -> Result<State> {
    let batches = engine.sql(&indoc::formatdoc! {"
        SELECT tick, ax, bx, ball_x, ball_y, vx, vy, score_a, score_b 
        FROM state 
        WHERE id = {STATE_ROW_ID} 
        LIMIT 1;
    "})?;
    let batch = first_batch(&batches)?;
    ensure_single_row(&batch)?;
    Ok(State {
        ax: value_at(&batch, 1),
        bx: value_at(&batch, 2),
        ball_x: value_at(&batch, 3),
        ball_y: value_at(&batch, 4),
        vx: value_at(&batch, 5),
        score_a: value_at(&batch, 7),
        score_b: value_at(&batch, 8),
    })
}

fn first_batch(batches: &[RecordBatch]) -> Result<RecordBatch> {
    batches
        .first()
        .cloned()
        .ok_or(Error::QueryReturnedNoBatches)
}

fn ensure_single_row(batch: &RecordBatch) -> Result<()> {
    if batch.num_rows() != 1 {
        return Err(Error::UnexpectedRowCount);
    }
    Ok(())
}

fn value_at(batch: &RecordBatch, index: usize) -> i64 {
    let column = batch.column(index);
    let array = column
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("expected Int64Array");
    array.value(0)
}

fn build_field_lines(params: &Params, state: &State) -> Vec<String> {
    let mut lines = Vec::with_capacity(params.h as usize);
    for y in 0..params.h {
        let mut row = String::with_capacity(params.w as usize);
        for x in 0..params.w {
            let ch = if y == 0 || y == params.h - 1 {
                HALF_BLOCK
            } else if (x == 1 && y >= state.ax && y < state.ax + params.paddle_h)
                || (x == params.w - 2 && y >= state.bx && y < state.bx + params.paddle_h)
                || (x == state.ball_x && y == state.ball_y)
                || (x == params.w / 2 && y % 3 == 1)
            {
                FULL_BLOCK
            } else {
                ' '
            };
            row.push(ch);
        }
        lines.push(row);
    }
    lines
}

fn fetch_digit_patterns(engine: &SqlEngine, digit: usize) -> Result<Vec<String>> {
    if digit > 9 {
        return Ok(vec![]);
    }
    let batches = engine.sql(&indoc::formatdoc! {"
        SELECT pattern 
        FROM digits 
        WHERE digit = {digit} 
        ORDER BY row;
    "})?;
    let batch = first_batch(&batches)?;

    let mut patterns = Vec::new();
    let column = batch.column(0);
    let array = column
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("expected StringArray");

    for i in 0..batch.num_rows() {
        if !array.is_null(i) {
            patterns.push(array.value(i).to_string());
        }
    }
    Ok(patterns)
}

fn draw_digit(
    engine: &SqlEngine,
    buf: &mut Vec<HudCell>,
    digit: usize,
    y: u16,
    x: u16,
    color: Color,
) -> Result<()> {
    let patterns = fetch_digit_patterns(engine, digit)?;
    for (i, pattern) in patterns.iter().enumerate() {
        let text = pattern.replace('F', &FULL_BLOCK.to_string());
        buf.push(HudCell {
            x,
            y: y + i as u16,
            text,
            color,
            bold: true,
            persist: true,
        });
    }
    Ok(())
}

fn render_frame(
    engine: &SqlEngine,
    params: &Params,
    state: &State,
    fps: u32,
    sound_enabled: bool,
    max_mode: bool,
    perf: &PerfStats,
) -> Result<()> {
    let lines = build_field_lines(params, state);
    let mut hud = Vec::new();

    let score_a = state.score_a.max(0) as u32;
    let score_b = state.score_b.max(0) as u32;

    let score_a_str = score_a.to_string();
    let score_b_str = score_b.to_string();

    let start_a = 38i32 - (score_a_str.len() as i32 * 4 - 1);
    for (idx, ch) in score_a_str.chars().enumerate() {
        let digit = ch.to_digit(10).unwrap() as usize;
        draw_digit(
            engine,
            &mut hud,
            digit,
            1,
            (start_a + (idx as i32 * 4)) as u16,
            Color::DarkGrey,
        )?;
    }

    for (idx, ch) in score_b_str.chars().enumerate() {
        let digit = ch.to_digit(10).unwrap() as usize;
        draw_digit(
            engine,
            &mut hud,
            digit,
            1,
            43 + (idx as u16 * 4),
            Color::DarkGrey,
        )?;
    }

    hud.push(HudCell {
        x: 0,
        y: 25,
        text: APP_TITLE.to_string(),
        color: Color::Yellow,
        bold: false,
        persist: false,
    });

    let actual_fps = perf.fps();
    let status = format!(
        "Press ESC to exit, S for sound [{}], +/- for framerate [{}] | actual {:.1} fps",
        if sound_enabled { "ON" } else { "OFF" },
        if max_mode {
            format!("{:.0} fps MAX", actual_fps.max(1.0))
        } else {
            format!("{} fps", fps)
        },
        actual_fps
    );

    hud.push(HudCell {
        x: 0,
        y: 26,
        text: status,
        color: Color::DarkGrey,
        bold: false,
        persist: false,
    });

    let perf_line = format!(
        "Frame {:.1} ms | Update {:.1} ms | Render {:.1} ms",
        perf.frame_ms(),
        perf.update_ms(),
        perf.render_ms(),
    );

    hud.push(HudCell {
        x: 0,
        y: 27,
        text: perf_line,
        color: Color::DarkGrey,
        bold: false,
        persist: false,
    });

    let stdout = stdout();
    let mut out = stdout.lock();

    for (row_idx, line) in lines.iter().enumerate() {
        queue!(out, MoveTo(0, row_idx as u16))?;

        let mut segment = String::new();
        let mut segment_color = Color::Reset;
        let mut segment_bold = false;
        let mut segment_initialized = false;
        let mut current_color = Color::Reset;
        let mut current_bold = false;

        for (col, ch) in line.chars().enumerate() {
            let col = col as u16;
            let (color, bold) = if ch == HALF_BLOCK {
                (Color::DarkGrey, false)
            } else if ch == FULL_BLOCK {
                if col == params.w as u16 / 2 {
                    (Color::DarkGrey, false)
                } else {
                    (Color::White, true)
                }
            } else {
                (Color::Reset, false)
            };

            if !segment_initialized {
                segment_initialized = true;
                segment_color = color;
                segment_bold = bold;
            } else if color != segment_color || bold != segment_bold {
                if !segment.is_empty() {
                    apply_style(
                        &mut out,
                        segment_color,
                        &mut current_color,
                        segment_bold,
                        &mut current_bold,
                    )?;
                    queue!(out, Print(&segment))?;
                    segment.clear();
                }
                segment_color = color;
                segment_bold = bold;
            }

            segment.push(ch);
        }

        if !segment.is_empty() {
            apply_style(
                &mut out,
                segment_color,
                &mut current_color,
                segment_bold,
                &mut current_bold,
            )?;
            queue!(out, Print(&segment))?;
        }

        if current_bold {
            queue!(out, SetAttribute(Attribute::NoBold))?;
        }
        if current_color != Color::Reset {
            queue!(out, ResetColor)?;
        }
    }

    let mut draws: Vec<HudCell> = Vec::new();
    let mut clears: Vec<(u16, u16, u16)> = Vec::new();

    HUD_CACHE.with(|state| {
        let mut cache = state.borrow_mut();
        let mut new_cache = FxHashMap::default();

        for cell in &hud {
            let width = cell.text.chars().count();
            let force_draw = cell.persist;
            match cache.remove(&(cell.x, cell.y)) {
                Some(prev) => {
                    if force_draw || prev.text != cell.text {
                        draws.push(cell.clone());
                    }
                    if prev.width > width {
                        clears.push((cell.x + width as u16, cell.y, (prev.width - width) as u16));
                    }
                    new_cache.insert(
                        (cell.x, cell.y),
                        HudCacheEntry {
                            text: cell.text.clone(),
                            width,
                        },
                    );
                }
                None => {
                    draws.push(cell.clone());
                    new_cache.insert(
                        (cell.x, cell.y),
                        HudCacheEntry {
                            text: cell.text.clone(),
                            width,
                        },
                    );
                }
            }
        }

        for ((x, y), prev) in cache.drain() {
            clears.push((x, y, prev.width as u16));
        }

        *cache = new_cache;
    });

    queue!(out, ResetColor, SetAttribute(Attribute::NoBold))?;

    for (x, y, width) in clears {
        if width == 0 {
            continue;
        }
        queue!(out, MoveTo(x, y), Print(" ".repeat(width as usize)))?;
    }

    for cell in draws {
        queue!(out, MoveTo(cell.x, cell.y))?;
        if cell.bold {
            queue!(out, SetAttribute(Attribute::Bold))?;
        } else {
            queue!(out, SetAttribute(Attribute::NoBold))?;
        }
        if cell.color == Color::Reset {
            queue!(out, ResetColor)?;
        } else {
            queue!(out, SetForegroundColor(cell.color))?;
        }
        queue!(out, Print(&cell.text))?;
        if cell.bold {
            queue!(out, SetAttribute(Attribute::NoBold))?;
        }
        if cell.color != Color::Reset {
            queue!(out, ResetColor)?;
        }
    }

    out.flush()?;
    Ok(())
}

fn apply_style<W>(
    out: &mut W,
    target_color: Color,
    current_color: &mut Color,
    target_bold: bool,
    current_bold: &mut bool,
) -> std::io::Result<()>
where
    W: Write + QueueableCommand,
{
    if *current_bold != target_bold {
        let attr = if target_bold {
            Attribute::Bold
        } else {
            Attribute::NoBold
        };
        queue!(out, SetAttribute(attr))?;
        *current_bold = target_bold;
    }

    if *current_color != target_color {
        if target_color == Color::Reset {
            queue!(out, ResetColor)?;
        } else {
            queue!(out, SetForegroundColor(target_color))?;
        }
        *current_color = target_color;
    }

    Ok(())
}

fn beep() {
    print!("\x07");
    let _ = stdout().flush();
}
