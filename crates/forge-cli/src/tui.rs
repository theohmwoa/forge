//! Terminal UI for browsing forge runs and diffs.
//!
//! Two modes:
//! - `view_run(steps)` — single-run timeline, content pane.
//! - `view_diff(a, b)` — side-by-side aligned diff using the same LCS as
//!   `forge diff`.

use std::io;
use std::time::Duration;

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use forge_core::{Step, StepKind};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Terminal,
};

use forge::{diff_chains, AlignedStep, DiffResult};

/// RAII guard so the terminal is restored even on panic / early return.
struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> anyhow::Result<Self> {
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture);
    }
}

pub fn view_run(steps: Vec<Step>, head_label: String) -> anyhow::Result<()> {
    let _guard = TerminalGuard::enter()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    let mut selected: usize = 0;

    loop {
        terminal.draw(|f| draw_single(f, &steps, &head_label, selected))?;
        if event::poll(Duration::from_millis(200))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Up | KeyCode::Char('k') => {
                        selected = selected.saturating_sub(1);
                    }
                    KeyCode::Down | KeyCode::Char('j') if selected + 1 < steps.len() => {
                        selected += 1;
                    }
                    KeyCode::Char('g') => selected = 0,
                    KeyCode::Char('G') => {
                        selected = steps.len().saturating_sub(1);
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

pub fn view_diff(
    a: Vec<Step>,
    b: Vec<Step>,
    a_label: String,
    b_label: String,
) -> anyhow::Result<()> {
    let diff = diff_chains(&a, &b);
    let entries = build_diff_entries(&a, &b, &diff);

    let _guard = TerminalGuard::enter()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    let mut selected: usize = 0;

    loop {
        terminal.draw(|f| draw_diff(f, &entries, &a_label, &b_label, selected))?;
        if event::poll(Duration::from_millis(200))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Up | KeyCode::Char('k') => {
                        selected = selected.saturating_sub(1);
                    }
                    KeyCode::Down | KeyCode::Char('j') if selected + 1 < entries.len() => {
                        selected += 1;
                    }
                    KeyCode::Char('g') => selected = 0,
                    KeyCode::Char('G') => {
                        selected = entries.len().saturating_sub(1);
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

fn draw_single(f: &mut ratatui::Frame<'_>, steps: &[Step], head: &str, selected: usize) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(f.area());

    let items: Vec<ListItem> = steps
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let marker = if i == selected { "▶ " } else { "  " };
            let short = short(&s.id.0);
            let label = label(&s.kind);
            let line = format!("{marker}{short}  {label}");
            let mut style = kind_style(&s.kind);
            if i == selected {
                style = style.add_modifier(Modifier::BOLD | Modifier::REVERSED);
            }
            ListItem::new(Line::from(Span::styled(line, style)))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" run {head}  ({} steps) ", steps.len())),
    );
    f.render_widget(list, chunks[0]);

    let content = steps
        .get(selected)
        .map(|s| format_content(&s.kind))
        .unwrap_or_default();
    let kind_label = steps
        .get(selected)
        .map(|s| label(&s.kind))
        .unwrap_or_default();
    let p = Paragraph::new(content).wrap(Wrap { trim: false }).block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" content — {kind_label} ")),
    );
    f.render_widget(p, chunks[1]);

    // Footer hint
    let footer = Paragraph::new(" ↑/↓ or j/k navigate · g/G top/bottom · q quit ")
        .style(Style::default().fg(Color::DarkGray));
    let footer_area = ratatui::layout::Rect {
        x: f.area().x,
        y: f.area().y + f.area().height.saturating_sub(1),
        width: f.area().width,
        height: 1,
    };
    f.render_widget(footer, footer_area);
}

#[derive(Clone)]
enum DiffEntry {
    SharedPrefix { step: Step },
    Aligned(AlignedStep),
}

fn build_diff_entries(a: &[Step], _b: &[Step], diff: &DiffResult) -> Vec<DiffEntry> {
    let mut out: Vec<DiffEntry> = Vec::new();
    for s in a.iter().take(diff.common_prefix_len) {
        out.push(DiffEntry::SharedPrefix { step: s.clone() });
    }
    for entry in &diff.aligned {
        out.push(DiffEntry::Aligned(entry.clone()));
    }
    out
}

fn draw_diff(
    f: &mut ratatui::Frame<'_>,
    entries: &[DiffEntry],
    a_label: &str,
    b_label: &str,
    selected: usize,
) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(f.area());

    let items: Vec<ListItem> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let (marker, label_text, style) = match e {
                DiffEntry::SharedPrefix { step } => (
                    "= ",
                    format!("{}  {}", short(&step.id.0), label(&step.kind)),
                    Style::default().fg(Color::DarkGray),
                ),
                DiffEntry::Aligned(AlignedStep::Match(s, _)) => (
                    "= ",
                    format!("{}  {}", short(&s.id.0), label(&s.kind)),
                    Style::default().fg(Color::DarkGray),
                ),
                DiffEntry::Aligned(AlignedStep::Modified(a, _)) => {
                    ("~ ", label(&a.kind), Style::default().fg(Color::Yellow))
                }
                DiffEntry::Aligned(AlignedStep::OnlyA(s)) => {
                    ("- ", label(&s.kind), Style::default().fg(Color::Red))
                }
                DiffEntry::Aligned(AlignedStep::OnlyB(s)) => {
                    ("+ ", label(&s.kind), Style::default().fg(Color::Green))
                }
            };
            let cursor = if i == selected { "▶" } else { " " };
            let line = format!("{cursor}{marker}{label_text}");
            let mut s = style;
            if i == selected {
                s = s.add_modifier(Modifier::BOLD | Modifier::REVERSED);
            }
            ListItem::new(Line::from(Span::styled(line, s)))
        })
        .collect();
    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" diff: A={a_label} vs B={b_label} ")),
    );
    f.render_widget(list, chunks[0]);

    // Right pane: detail of the selected entry. For Modified, stack A and B.
    let lines = entries.get(selected).map(detail_lines).unwrap_or_default();
    let p = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title(" content "));
    f.render_widget(p, chunks[1]);

    let footer = Paragraph::new(
        " ↑/↓ or j/k navigate · g/G top/bottom · q quit · legend: =same ~mod -A +B ",
    )
    .style(Style::default().fg(Color::DarkGray));
    let footer_area = ratatui::layout::Rect {
        x: f.area().x,
        y: f.area().y + f.area().height.saturating_sub(1),
        width: f.area().width,
        height: 1,
    };
    f.render_widget(footer, footer_area);
}

fn detail_lines(entry: &DiffEntry) -> Vec<Line<'static>> {
    match entry {
        DiffEntry::SharedPrefix { step } | DiffEntry::Aligned(AlignedStep::Match(step, _)) => {
            let mut out = vec![Line::from(Span::styled(
                format!("[shared] {}", label(&step.kind)),
                Style::default().fg(Color::DarkGray),
            ))];
            out.extend(text_lines(&format_content(&step.kind), Color::White));
            out
        }
        DiffEntry::Aligned(AlignedStep::Modified(a, b)) => {
            let mut out = vec![Line::from(Span::styled(
                format!("[modified] {}", label(&a.kind)),
                Style::default().fg(Color::Yellow),
            ))];
            out.push(Line::from(Span::styled(
                "── A ─────",
                Style::default().fg(Color::Red),
            )));
            out.extend(text_lines(&format_content(&a.kind), Color::Red));
            out.push(Line::from(""));
            out.push(Line::from(Span::styled(
                "── B ─────",
                Style::default().fg(Color::Green),
            )));
            out.extend(text_lines(&format_content(&b.kind), Color::Green));
            out
        }
        DiffEntry::Aligned(AlignedStep::OnlyA(s)) => {
            let mut out = vec![Line::from(Span::styled(
                format!("[only A] {}", label(&s.kind)),
                Style::default().fg(Color::Red),
            ))];
            out.extend(text_lines(&format_content(&s.kind), Color::Red));
            out
        }
        DiffEntry::Aligned(AlignedStep::OnlyB(s)) => {
            let mut out = vec![Line::from(Span::styled(
                format!("[only B] {}", label(&s.kind)),
                Style::default().fg(Color::Green),
            ))];
            out.extend(text_lines(&format_content(&s.kind), Color::Green));
            out
        }
    }
}

fn text_lines(s: &str, color: Color) -> Vec<Line<'static>> {
    s.lines()
        .map(|l| Line::from(Span::styled(l.to_string(), Style::default().fg(color))))
        .collect()
}

fn kind_style(kind: &StepKind) -> Style {
    match kind {
        StepKind::Prompt { .. } => Style::default().fg(Color::Yellow),
        StepKind::Message { role, .. } if role == "assistant" => Style::default().fg(Color::Cyan),
        StepKind::Message { .. } => Style::default().fg(Color::White),
        StepKind::ToolCall { .. } => Style::default().fg(Color::Magenta),
        StepKind::ToolResult { .. } => Style::default().fg(Color::Green),
    }
}

fn label(kind: &StepKind) -> String {
    match kind {
        StepKind::Prompt { model, .. } => format!("prompt[{model}]"),
        StepKind::Message { role, .. } => format!("message[{role}]"),
        StepKind::ToolCall { name, .. } => format!("tool_call[{name}]"),
        StepKind::ToolResult { call_id, .. } => format!("tool_result[{call_id}]"),
    }
}

fn format_content(kind: &StepKind) -> String {
    match kind {
        StepKind::Prompt { content, .. } => content.clone(),
        StepKind::Message { content, .. } => content.clone(),
        StepKind::ToolCall { input, .. } => {
            serde_json::to_string_pretty(input).unwrap_or_else(|_| input.to_string())
        }
        StepKind::ToolResult { output, .. } => {
            serde_json::to_string_pretty(output).unwrap_or_else(|_| output.to_string())
        }
    }
}

fn short(s: &str) -> String {
    s.chars().take(10).collect()
}
