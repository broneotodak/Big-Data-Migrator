-- Initial database schema for Big Data Migrator

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- User Sessions
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    session_data JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT session_data_size CHECK (length(session_data::text) <= 10000)
);

-- File Metadata
CREATE TABLE file_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES user_sessions(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    size BIGINT NOT NULL,
    processing_status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT valid_processing_status CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed'))
);

-- Processing States
CREATE TABLE processing_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_id UUID NOT NULL REFERENCES file_metadata(id) ON DELETE CASCADE,
    current_step TEXT NOT NULL,
    progress_percentage FLOAT NOT NULL DEFAULT 0.0,
    error_log TEXT[] DEFAULT ARRAY[]::TEXT[],
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    state_data JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT valid_progress CHECK (progress_percentage >= 0.0 AND progress_percentage <= 100.0)
);

-- Conversation History
CREATE TABLE conversation_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES user_sessions(id) ON DELETE CASCADE,
    message_type TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    llm_response TEXT,
    context_data JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT valid_message_type CHECK (message_type IN ('user', 'assistant', 'system', 'error'))
);

-- Data Insights
CREATE TABLE data_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_id UUID NOT NULL REFERENCES file_metadata(id) ON DELETE CASCADE,
    column_analysis JSONB DEFAULT '{}'::jsonb,
    relationships JSONB DEFAULT '[]'::jsonb,
    quality_score FLOAT NOT NULL DEFAULT 0.0,
    recommendations TEXT[] DEFAULT ARRAY[]::TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_quality_score CHECK (quality_score >= 0.0 AND quality_score <= 100.0)
);

-- Resource Usage
CREATE TABLE resource_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES user_sessions(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    cpu_percent FLOAT NOT NULL,
    memory_percent FLOAT NOT NULL,
    memory_used_mb FLOAT NOT NULL,
    disk_usage_percent FLOAT NOT NULL,
    operation_type TEXT NOT NULL,
    operation_id UUID,
    CONSTRAINT valid_percentages CHECK (
        cpu_percent >= 0.0 AND cpu_percent <= 100.0 AND
        memory_percent >= 0.0 AND memory_percent <= 100.0 AND
        disk_usage_percent >= 0.0 AND disk_usage_percent <= 100.0
    )
);

-- Indexes
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_last_active ON user_sessions(last_active);
CREATE INDEX idx_file_metadata_session_id ON file_metadata(session_id);
CREATE INDEX idx_file_metadata_processing_status ON file_metadata(processing_status);
CREATE INDEX idx_processing_states_file_id ON processing_states(file_id);
CREATE INDEX idx_conversation_history_session_id ON conversation_history(session_id);
CREATE INDEX idx_conversation_history_timestamp ON conversation_history(timestamp);
CREATE INDEX idx_data_insights_file_id ON data_insights(file_id);
CREATE INDEX idx_resource_usage_session_id ON resource_usage(session_id);
CREATE INDEX idx_resource_usage_timestamp ON resource_usage(timestamp);

-- Functions
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers
CREATE TRIGGER update_file_metadata_updated_at
    BEFORE UPDATE ON file_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_data_insights_updated_at
    BEFORE UPDATE ON data_insights
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Stored Procedures
CREATE OR REPLACE FUNCTION cleanup_old_sessions(cutoff_date TIMESTAMP WITH TIME ZONE)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    WITH deleted AS (
        DELETE FROM user_sessions
        WHERE last_active < cutoff_date
        RETURNING id
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION cleanup_old_conversations(cutoff_date TIMESTAMP WITH TIME ZONE)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    WITH deleted AS (
        DELETE FROM conversation_history
        WHERE timestamp < cutoff_date
        RETURNING id
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION cleanup_old_resource_usage(cutoff_date TIMESTAMP WITH TIME ZONE)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    WITH deleted AS (
        DELETE FROM resource_usage
        WHERE timestamp < cutoff_date
        RETURNING id
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql; 