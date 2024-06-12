use crossterm::event::{Event, KeyCode, KeyEventKind};

/// Takes an event, checks if it is a key press event, and returns the [`KeyCode`]
pub(super) fn event_keycode(event: &Event) -> Option<KeyCode> {
    let Event::Key(key) = event else {
        return None;
    };

    if key.kind != KeyEventKind::Press {
        return None;
    }

    Some(key.code)
}
