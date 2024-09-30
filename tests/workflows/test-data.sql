-- Insert Users
INSERT INTO User (id, email, name) VALUES ('user1_id', 'user1@example.com', 'User 1');
INSERT INTO User (id, email, name) VALUES ('user2_id', 'user2@example.com', 'User 2');
INSERT INTO User (id, email, name) VALUES ('user3_id', 'user3@example.com', 'User 3');
INSERT INTO User (id, email, name) VALUES ('user4_id', 'user4@example.com', 'User 4');
INSERT INTO User (id, email, name) VALUES ('user5_id', 'user5@example.com', 'User 5');

-- Insert Models
INSERT INTO Model (id, name, maxLength, tokenLimit) VALUES ('model1_id', 'GPT-4', 2048, 4096);
INSERT INTO Model (id, name, maxLength, tokenLimit) VALUES ('model2_id', 'GPT-3', 1024, 2048);

-- Insert Folders
INSERT INTO Folder (id, userId, name, isRoot) VALUES ('folder1_id', 'user1_id', 'User 1 Folder', true);
INSERT INTO Folder (id, userId, name, isRoot) VALUES ('folder2_id', 'user2_id', 'User 2 Folder', true);
INSERT INTO Folder (id, userId, name, isRoot) VALUES ('folder3_id', 'user3_id', 'User 3 Folder', true);
INSERT INTO Folder (id, userId, name, isRoot) VALUES ('folder4_id', 'user4_id', 'User 4 Folder', true);
INSERT INTO Folder (id, userId, name, isRoot) VALUES ('folder5_id', 'user5_id', 'User 5 Folder', true);

-- Insert Conversations
INSERT INTO Conversation (id, userEmail, temperature, folderId, modelId, name, prompt) VALUES
('conv1_id', 'user1@example.com', 0, 'folder1_id', 'model1_id', 'Conversation 1', 'Prompt for conversation 1'),
('conv2_id', 'user1@example.com', 1, 'folder1_id', 'model2_id', 'Conversation 2', 'Prompt for conversation 2'),
('conv3_id', 'user2@example.com', 2, 'folder2_id', 'model1_id', 'Conversation 3', 'Prompt for conversation 3'),
('conv4_id', 'user2@example.com', 0, 'folder2_id', 'model2_id', 'Conversation 4', 'Prompt for conversation 4'),
('conv5_id', 'user3@example.com', 1, 'folder3_id', 'model1_id', 'Conversation 5', 'Prompt for conversation 5'),
('conv6_id', 'user3@example.com', 0, 'folder3_id', 'model2_id', 'Conversation 6', 'Prompt for conversation 6'),
('conv7_id', 'user4@example.com', 2, 'folder4_id', 'model1_id', 'Conversation 7', 'Prompt for conversation 7'),
('conv8_id', 'user4@example.com', 1, 'folder4_id', 'model2_id', 'Conversation 8', 'Prompt for conversation 8'),
('conv9_id', 'user5@example.com', 0, 'folder5_id', 'model1_id', 'Conversation 9', 'Prompt for conversation 9'),
('conv10_id', 'user5@example.com', 1, 'folder5_id', 'model2_id', 'Conversation 10', 'Prompt for conversation 10');

-- Insert Messages
INSERT INTO Message (id, role, content, createdAt, conversationId) VALUES
('msg1_id', 'system', 'Welcome to the conversation!', '2024-09-25 10:00:00', 'conv1_id'),
('msg2_id', 'user', 'This is a user message.', '2024-09-25 10:01:00', 'conv1_id'),
('msg3_id', 'system', 'Conversation started.', '2024-09-25 11:00:00', 'conv2_id'),
('msg4_id', 'user', 'Another message from user.', '2024-09-25 11:02:00', 'conv2_id'),
('msg5_id', 'system', 'Welcome to the conversation!', '2024-09-25 09:00:00', 'conv3_id'),
('msg6_id', 'user', 'User input here.', '2024-09-25 09:05:00', 'conv3_id'),
('msg7_id', 'system', 'Conversation initialized.', '2024-09-25 12:00:00', 'conv4_id'),
('msg8_id', 'user', 'User message in conversation 4.', '2024-09-25 12:05:00', 'conv4_id'),
('msg9_id', 'system', 'Chat started.', '2024-09-25 13:00:00', 'conv5_id'),
('msg10_id', 'user', 'User feedback here.', '2024-09-25 13:10:00', 'conv5_id');
