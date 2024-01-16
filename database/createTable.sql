CREATE TABLE log (
    id INTEGER PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    numero_etape INTEGER,
    login TEXT,
    commentaire TEXT,
    numero_badge TEXT
);
