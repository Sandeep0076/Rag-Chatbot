-- Insert Users
INSERT INTO User (id, email, name) VALUES ('user1_id', 'user1@example.com', 'User 1');
INSERT INTO User (id, email, name) VALUES ('user2_id', 'user2@example.com', 'User 2');
INSERT INTO User (id, email, name) VALUES ('user3_id', 'user3@example.com', 'User 3');
-- user marked as deletion candidate, but should not be deleted because timestamp is lower than 4 week
INSERT INTO User (id, email, name, wf_deletion_candidate, wf_deletion_timestamp) VALUES ('user4_id', 'user4@example.com', 'User 4', True, TIMESTAMP_LESS_THAN_4_WEEKS);
INSERT INTO User (id, email, name, wf_deletion_candidate, wf_deletion_timestamp) VALUES ('user7_id', 'user8@example.com', 'User 8', True, TIMESTAMP_LESS_THAN_4_WEEKS);
INSERT INTO User (id, email, name, wf_deletion_candidate, wf_deletion_timestamp) VALUES ('user8_id', 'user7@example.com', 'User 7', True, TIMESTAMP_LESS_THAN_4_WEEKS);
-- user marked and timestamp older than 4 weeks
INSERT INTO User (id, email, name, wf_deletion_candidate, wf_deletion_timestamp) VALUES ('user5_id', 'user5@example.com', 'User 5', True, TIMESTAMP_MORE_THAN_4_WEEKS);
INSERT INTO User (id, email, name, wf_deletion_candidate, wf_deletion_timestamp) VALUES ('user6_id', 'user6@example.com', 'User 6', True, TIMESTAMP_MORE_THAN_4_WEEKS);
-- operational user to be replaced for shared prompts, should be excluded from workflow in all phases
INSERT INTO User (id, email, name) VALUES ('dXNlci5kZWxldGVkQHJ0bC5kZQo=', 'user.deleted@rtl.de', 'Nutzer, Gelöschter [RTL Tech]');

-- Insert Models
INSERT INTO Model (id, name, maxLength, tokenLimit) VALUES ('model1_id', 'GPT-4', 2048, 4096);
INSERT INTO Model (id, name, maxLength, tokenLimit) VALUES ('model2_id', 'GPT-3', 1024, 2048);

-- Insert Folders
INSERT INTO Folder (id, userId, name, isRoot) VALUES ('folder1_id', 'user1_id', 'User 1 Folder', true);
INSERT INTO Folder (id, userId, name, isRoot) VALUES ('folder2_id', 'user1_id', 'User 1 Folder 2', true);
INSERT INTO Folder (id, userId, name, isRoot) VALUES ('folder3_id', 'user3_id', 'User 3 Folder', true);
INSERT INTO Folder (id, userId, name, isRoot) VALUES ('folder4_id', 'user4_id', 'User 4 Folder', true);
INSERT INTO Folder (id, userId, name, isRoot) VALUES ('folder5_id', 'user5_id', 'User 5 Folder', true);
INSERT INTO Folder (id, userId, name, isRoot) VALUES ('folder6_id', 'user5_id', 'User 5 Folder 2', true);
INSERT INTO Folder (id, userId, name, isRoot) VALUES ('folder7_id', 'user7_id', 'User 7 Folder 1', true);
INSERT INTO Folder (id, userId, name, isRoot) VALUES ('folder8_id', 'user8_id', 'User 8 Folder 1', true);

-- Insert Conversations
INSERT INTO Conversation (id, userEmail, temperature, folderId, modelId, name, prompt, fileId) VALUES
('conv1_id', 'user1@example.com', 0, 'folder1_id', 'model1_id', 'Conversation 1', 'Prompt for conversation 1', '1dd20782-0e60-4f17-90a1-dde343786de4'),
('conv2_id', 'user1@example.com', 1, 'folder1_id', 'model2_id', 'Conversation 2', 'Prompt for conversation 2', NULL),
('conv3_id', 'user1@example.com', 2, 'folder2_id', 'model1_id', 'Conversation 3', 'Prompt for conversation 3', NULL),
('conv4_id', 'user2@example.com', 0, 'folder2_id', 'model2_id', 'Conversation 4', 'Prompt for conversation 4', NULL),
('conv5_id', 'user3@example.com', 1, 'folder3_id', 'model1_id', 'Conversation 5', 'Prompt for conversation 5', '57a1121f-a5ab-4d97-a73c-52a971056242'),
('conv6_id', 'user3@example.com', 0, 'folder3_id', 'model2_id', 'Conversation 6', 'Prompt for conversation 6', 'f4d17237-a76f-437c-8230-79979b6f7d62'),
('conv7_id', 'user3@example.com', 2, 'folder4_id', 'model1_id', 'Conversation 7', 'Prompt for conversation 7', NULL),
('conv8_id', 'user3@example.com', 1, 'folder4_id', 'model2_id', 'Conversation 8', 'Prompt for conversation 8', NULL),
('conv9_id', 'user5@example.com', 0, 'folder5_id', 'model1_id', 'Conversation 9', 'Prompt for conversation 9', '767703e0-8195-4345-914b-81bbaa0588b7'),
('conv10_id', 'user5@example.com', 1, 'folder5_id', 'model2_id', 'Conversation 10', 'Prompt for conversation 10', '4345914b-a034-4a43-bee4-df5c0f0208fb'),
('conv11_id', 'user7@example.com', 1, 'folder7_id', 'model2_id', 'Conversation 11', 'Prompt for conversation 11', '819532e6-a034-4a43-bee4-df5c0f0208fb'),
('conv12_id', 'user8@example.com', 1, 'folder8_id', 'model2_id', 'Conversation 12', 'Prompt for conversation 12', '914b32e6-4a43-bee4-bee4-81bb0f0208fb');

-- Insert Messages
INSERT INTO Message (id, role, content, createdAt, conversationId) VALUES
('msg1_id', 'system', 'Welcome to the conversation!', '2024-09-25 10:00:00', 'conv1_id'),
('msg2_id', 'user', 'This is a user message.', '2024-09-25 10:01:00', 'conv1_id'),
('msg3_id', 'system', 'Conversation started.', '2024-09-25 11:00:00', 'conv2_id'),
('msg4_id', 'user', 'Another message from user.', '2024-09-25 11:02:00', 'conv2_id'),
('msg5_id', 'system', 'Welcome to the conversation!', '2024-09-25 09:00:00', 'conv3_id'),
('msg6_id', 'user', 'User input here.', '2024-09-25 09:05:00', 'conv3_id'),
('msg7_id', 'system', 'Conversation initialized.', '2024-09-25 12:00:00', 'conv4_id'),
('msg8_id', 'user', 'User message in conversation 4.', '2024-09-25 12:05:00', 'conv9_id'),
('msg9_id', 'system', 'Chat started.', '2024-09-25 13:00:00', 'conv9_id'),
('msg10_id', 'user', 'User feedback here.', '2024-09-25 13:10:00', 'conv10_id'),
('msg11_id', 'system', 'User feedback here.', '2024-09-25 13:10:00', 'conv11_id'),
('msg12_id', 'user', 'User feedback here.', '2024-09-25 13:10:00', 'conv11_id'),
('msg13_id', 'system', 'User feedback here.', '2024-09-25 13:10:00', 'conv11_id'),
('msg14_id', 'user', 'User feedback here.', '2024-09-25 13:10:00', 'conv12_id'),
('msg15_id', 'system', 'User feedback here.', '2024-09-25 13:10:00', 'conv13_id');
