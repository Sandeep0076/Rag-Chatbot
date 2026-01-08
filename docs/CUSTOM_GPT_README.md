# Custom GPT Creation Guide

## Overview

Custom GPT enables you to create personalized AI assistants tailored to your specific needs. Through an intuitive guided process, you can design GPTs with custom personalities, knowledge domains, and communication styles - all without requiring technical expertise.

Unlike generic AI assistants, Custom GPTs are purpose-built for your unique requirements, whether you need a coding tutor, customer service agent, creative writing assistant, or specialized consultant.

## Key Features

### 🎯 Guided Creation Process
A step-by-step wizard walks you through defining your GPT without overwhelming technical details. The system asks thoughtful questions and provides examples to guide your decisions.

### 🎨 Personality Customization
Choose from multiple communication styles:
- Professional & Concise
- Friendly & Encouraging  
- Casual & Conversational
- Technical & Detailed
- Witty & Engaging

### 📚 Knowledge Management
Optionally upload reference documents (PDF, Word, Excel, CSV, Text) that your GPT can reference during conversations. Your GPT works perfectly fine without documents using general knowledge.

### ⚙️ Flexible Configuration
Fine-tune your GPT with:
- Model selection (GPT-4, Gemini, and other available models)
- Temperature control (creativity vs. precision)
- Custom capabilities and boundaries
- Specialized terminology and jargon

### 🧠 Smart Answer Resolution
The system intelligently understands your responses. If you like an example provided by the assistant, you can simply say "I like the first one" or "Combine the first and second examples". The system will automatically resolve this reference into the full text before proceeding, ensuring your intent is captured accurately.

### 🔒 Safety & Control
Define what your GPT should and shouldn't do, ensuring appropriate guardrails for your use case.

## How to Use

### Creating Your First Custom GPT

#### Step 1: Access the Feature
Navigate to the **Custom GPT** tab in the application interface.

#### Step 2: Describe Your Idea
In the "Initial Idea" section, describe what you want your GPT to do. Examples:
- "I need a customer service assistant for my e-commerce store"
- "I want a coding tutor that explains Python concepts to beginners"
- "I need help creating professional email drafts"

#### Step 3: Answer Guided Questions
Work through 6 steps of thoughtful questions that guide you through:
- **Purpose Clarification**: Define specific problems your GPT should solve
- **Audience Understanding**: Identify who will use this GPT and their expertise level
- **Tone & Style**: Select the communication style that fits your use case
- **Capabilities Definition**: List what your GPT must do well and define boundaries
- **Knowledge & Context**: Specify specialized knowledge domains and terminology
- **Examples Collection**: Provide ideal interaction scenarios and custom instructions

**How It Works - Detailed Workflow:**

1. **Question Generation**: For each step, the system generates a contextual question with 2-3 example answers based on your previous responses.

2. **Your Response**: You can respond in three ways:
   - **Custom Answer**: Type your own answer (e.g., "I want to help beginners learn Python")
   - **Reference Examples**: Simply say "I like the first one" or "The second example" or "Combine examples 1 and 3"
   - **Skip**: Leave the field empty if you want to skip that particular question

3. **Smart Answer Resolution**: When you click "Next", the system intelligently processes your answer:
   - If you referenced an example (e.g., "The first one"), the system automatically resolves this into the full text of that example
   - If you combined examples (e.g., "Mix 1 and 2"), the system merges them intelligently
   - If you provided a custom answer, it's used as-is
   - If you skipped, an empty value is passed forward

4. **Context Building**: Your resolved answer is then used as context for generating the next question, ensuring each subsequent question is tailored to your specific needs.

5. **Progressive Refinement**: Each step builds on the previous ones, creating a comprehensive understanding of your Custom GPT requirements.

**Example Workflow:**
- **Step 1 (Purpose)**: System shows examples like "Help with customer support" and "Teach programming"
- **You say**: "The second one"
- **System resolves**: "Teach programming" (full text from example)
- **Step 2 (Audience)**: System generates question using "Teach programming" as context
- **You say**: "Complete beginners who have never coded before"
- **System uses**: Your custom answer directly
- **Step 3 (Tone)**: System generates question considering both teaching programming and beginner audience

Each step includes clear questions and examples. There are no wrong answers - be honest about your needs.

#### Step 4: Review Generated Prompt and Name
The system synthesizes all your inputs into:
- **GPT Name**: Auto-generated based on your configuration (2-5 words, in your detected language)
- **System Prompt**: Comprehensive instruction set that defines your GPT's behavior

Review both to ensure they capture your vision. You can edit the GPT name if desired.

#### Step 5: Configure Settings
On the settings page:
1. **Review/Edit GPT name** - Auto-populated from generation, but you can customize it
2. **Select a model** from available options (see Settings & Configuration section below for details)
3. **Adjust temperature** to control response creativity (see Settings & Configuration section below for recommended values)
4. **Upload documents** (optional):
   - Click "Upload reference documents"
   - Select files (PDF, DOCX, TXT, CSV, XLSX)
   - Click "Process Documents"
   - Documents appear in the "Currently Attached Documents" list
   - Remove documents individually if needed

#### Step 6: Start Chatting
Click "Start Chat" to launch your custom GPT. The chat interface will display:
- Your GPT's name in a distinctive banner
- List of attached documents (if any)
- Ready-to-use chat interface

### Using Your Custom GPT

Once activated, your Custom GPT follows its defined role and purpose consistently, references uploaded documents when relevant (citing sources), maintains its communication style throughout conversations, respects boundaries you've set during creation, and uses specialized knowledge as configured.

### Managing Documents

**Adding Documents:**
- Upload multiple files simultaneously
- Process them before starting your chat
- Each document becomes part of your GPT's knowledge base

**Removing Documents:**
- Click "Remove" next to any document in the settings page
- Changes take effect when you start a new chat

**Working Without Documents:**
Your Custom GPT functions perfectly without uploaded documents, using its general knowledge within the scope you defined.

## Settings & Configuration

### Model Selection
Choose from available models based on your needs:
- **GPT-4 models**: Advanced reasoning, complex tasks
- **Gemini models**: Multimodal capabilities, efficient processing
- **Mini models**: Faster responses, cost-effective

### Temperature Control
Adjust response variability:
- **0.0-0.3**: Highly deterministic, factual, consistent
- **0.4-0.7**: Balanced creativity and reliability (recommended for most use cases)
- **0.8-1.2**: More creative, varied responses
- **1.3-2.0**: Maximum creativity, experimental

### Document Requirements
Supported file formats:
- **PDF**: Research papers, manuals, reports
- **Word**: Documentation, guides, policies
- **Excel/CSV**: Data tables, spreadsheets
- **Text**: Plain text documents, transcripts

Maximum recommendations:
- File size: Best performance under 10MB per file
- Total files: No strict limit, but 5-10 documents recommended
- Content: Ensure documents are text-based and relevant

## Use Cases

### Business Applications
- **Customer Support Agent**: Handle common inquiries, troubleshoot issues, escalate complex cases
- **Sales Assistant**: Generate proposals, answer product questions, draft follow-up emails
- **HR Helper**: Assist with policy questions, onboarding guidance, benefits information

### Educational Applications
- **Subject Tutor**: Teach specific subjects (math, science, languages) with custom pedagogy
- **Coding Mentor**: Guide programming learners, explain errors, suggest best practices
- **Writing Coach**: Improve writing skills, provide feedback, suggest improvements

### Creative Applications
- **Content Writer**: Generate blog posts, social media content, marketing copy
- **Brainstorming Partner**: Ideation, creative solutions, alternative perspectives
- **Story Assistant**: Plot development, character creation, world-building

### Professional Services
- **Technical Consultant**: Domain-specific advice with uploaded documentation
- **Research Assistant**: Summarize papers, extract insights, answer domain questions
- **Legal/Medical Assistant**: Reference uploaded guidelines (not a replacement for professionals)

### Personal Productivity
- **Email Drafting**: Create professional correspondence quickly
- **Meeting Prep**: Summarize materials, generate agendas, suggest talking points
- **Learning Companion**: Study materials, quiz preparation, concept clarification

## Best Practices

### Creating Effective GPTs

**Be Specific About Purpose:**
- Define narrow, clear objectives rather than broad goals
- Example: "Help debug Python code" vs. "Teach programming"

**Choose Appropriate Tone:**
- Match communication style to your audience
- Technical users appreciate precision; beginners need encouragement

**Set Clear Boundaries:**
- Explicitly state what your GPT should NOT do
- Example: "Don't make medical diagnoses, only reference uploaded guidelines"

**Provide Good Examples:**
- Share realistic scenarios during creation
- Quality examples lead to better GPT behavior

**Upload Relevant Documents:**
- Only include documents your GPT genuinely needs
- More documents ≠ better performance; relevance matters

### Optimizing Performance

**Start Simple:**
- Create a basic version first
- Test and refine based on actual usage

**Iterate and Improve:**
- If responses aren't quite right, adjust temperature
- Consider creating a new version with refined instructions

**Test Thoroughly:**
- Try edge cases and unusual queries
- Ensure boundaries are respected

**Document Management:**
- Keep uploaded documents current and relevant
- Remove outdated materials

## Complete Workflow

### High-Level Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ Step 0: Initial Idea                                        │
│ - User enters initial idea                                  │
│ - System detects language                                   │
│ - System generates purpose clarification question           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Steps 1-6: Guided Configuration                             │
│                                                              │
│ Step 1: Purpose Clarification                               │
│   → User answers or skips                                   │
│   → System resolves answer (if references examples)         │
│   → System generates audience question                      │
│                                                              │
│ Step 2: Audience Understanding                              │
│   → User answers or skips                                   │
│   → System generates tone/style options                     │
│                                                              │
│ Step 3: Tone & Style                                        │
│   → User selects from predefined options or skips           │
│   → System generates capabilities question                  │
│                                                              │
│ Step 4: Capabilities Definition                             │
│   → User answers or skips                                   │
│   → System resolves answer                                  │
│   → System generates knowledge question                     │
│                                                              │
│ Step 5: Knowledge & Context                                 │
│   → User answers or skips                                   │
│   → System resolves answer                                  │
│   → System generates examples question                       │
│                                                              │
│ Step 6: Examples Collection                                 │
│   → User provides example interaction or skips             │
│   → System resolves answer                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 7: System Prompt & Name Generation                     │
│ - System synthesizes all inputs                             │
│ - Generates GPT name (auto, 2-5 words)                      │
│ - Generates comprehensive system prompt                     │
│ - User can review and edit both                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 8: Settings Configuration                              │
│ - GPT name auto-populated (user can edit)                   │
│ - User selects model                                        │
│ - User adjusts temperature                                  │
│ - User uploads documents (optional)                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 9: Start Chatting                                       │
│ - User clicks "Start Chat"                                  │
│ - System initializes chat session                           │
│ - Chat interface becomes available                           │
│                                                              │
│ Ongoing Chat:                                                │
│ - User sends messages                                        │
│ - System responds using custom GPT configuration             │
│ - Documents referenced when relevant (if uploaded)          │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Step-by-Step Workflow

#### Step 0: Initial Idea
1. User enters initial idea in text area
2. **Backend API Call**: `POST /language/detect` - Detects language
3. **Backend API Call**: `POST /chat/azure` - Generates purpose clarification question
4. System displays question with 2-3 example answers
5. User can proceed to Step 1

#### Steps 1-6: Configuration Loop
For each step:
1. System displays question and examples from previous API call
2. User provides answer (or clicks Skip)
3. **If user references examples**:
   - **Backend API Call**: `POST /chat/azure` (Answer Resolution)
   - System resolves reference to full text
4. **Backend API Call**: `POST /chat/azure` (Next Step Question)
5. System displays next question
6. Process repeats for next step

#### Step 7: System Prompt & Name Generation
1. After Step 6 completion, system automatically generates prompt and name
2. **Backend API Call**: `POST /gpt/generate-system-prompt`
3. System displays:
   - Auto-generated GPT name (2-5 words, in detected language)
   - Generated system prompt
4. User can edit and save changes to both
5. User can regenerate if needed

#### Step 8: Settings & Document Upload
1. GPT name auto-populated from Step 7 (user can edit)
2. User configures model and temperature
2. **Optional**: User uploads documents
   - **Backend API Call**: `POST /file/upload` (with `custom_gpt: true`)
   - System processes documents and stores file IDs
3. User reviews settings
4. User clicks "Start Chat"

#### Step 9: Chat Initialization
1. System determines chat endpoint:
   - **With documents**: `POST /file/chat`
   - **Without documents**: `POST /chat/anthropic`
2. System initializes chat session with:
   - Custom system prompt
   - Document IDs (if any)
   - Session ID
   - Model and temperature settings
3. Chat interface becomes active
4. User can start conversing with Custom GPT

### API Workflow Summary

**Total API Calls Required**:
- 1x Language Detection
- 6-12x Azure Chat (questions + optional resolutions)
- 1x System Prompt Generation
- 0-Nx File Upload (optional, can be batch)
- 1x Available Models (for settings)
- Ongoing: Chat API calls

**Key API Endpoints**:
- `POST /language/detect` - Language detection
- `POST /chat/azure` - Question generation & answer resolution
- `POST /gpt/generate-system-prompt` - System prompt generation
- `POST /file/upload` - Document upload (with `custom_gpt: true`)
- `GET /available-models` - Model selection
- `POST /file/chat` - Chat with documents
- `POST /chat/anthropic` - Chat without documents

For complete API documentation, see [Custom GPT API Integration Guide](./CUSTOM_GPT_API_INTEGRATION.md).

## Technical Requirements

### Frontend (User-Facing)
- **Platform**: Web-based Streamlit application
- **Access**: Available through Custom GPT tab
- **Browser**: Modern web browsers (Chrome, Firefox, Safari, Edge)
- **API Integration**: Requires backend API access with authentication

### Backend API Requirements
The Custom GPT feature requires the following backend APIs:

1. **Language Detection API** (`POST /language/detect`)
   - Detects language from user input
   - Returns language code (e.g., "English", "German")

2. **Azure Chat API** (`POST /chat/azure`)
   - Generates contextual questions for each configuration step
   - Resolves user input references to examples
   - Requires authentication

3. **System Prompt Generation API** (`POST /gpt/generate-system-prompt`)
   - Synthesizes all user inputs into system prompt
   - Validates and optimizes prompt generation

4. **File Upload API** (`POST /file/upload`)
   - Uploads reference documents
   - Supports batch upload for multiple files
   - Requires `custom_gpt: true` parameter
   - Returns file IDs and session ID

5. **Available Models API** (`GET /available-models`)
   - Returns list of available AI models
   - Includes model type categorization

6. **Chat APIs**
   - `POST /file/chat` - Chat with documents
   - `POST /chat/anthropic` - Chat without documents
   - Both support `custom_gpt: true` and `system_prompt` parameters

**Authentication**: Most endpoints require Bearer token authentication.

**Current Status**: Frontend implementation complete. Backend APIs must be configured with proper authentication and support for Custom GPT parameters.

For detailed API integration guide, see [Custom GPT API Integration Guide](./CUSTOM_GPT_API_INTEGRATION.md).

### Data Privacy
- Uploaded documents are processed and stored securely
- Document content used only within your custom GPT sessions
- No cross-user document sharing
- Authentication required for all API access

## Frequently Asked Questions

### General Questions

**Q: Do I need technical knowledge to create a Custom GPT?**  
A: No. The guided process uses plain language questions and examples.

**Q: Can I create multiple Custom GPTs?**  
A: Yes, you can create as many specialized GPTs as you need for different purposes.

**Q: How long does it take to create a Custom GPT?**  
A: Typically 5-10 minutes to complete all steps and configuration.

**Q: Can I modify my Custom GPT after creation?**  
A: Currently, you would create a new version with updated settings. Iterative improvement is straightforward.

### Document Questions

**Q: Are documents required?**  
A: No, they're completely optional. Your GPT works perfectly without documents.

**Q: What happens if I upload documents?**  
A: Your GPT prioritizes document content when answering relevant questions and cites sources.

**Q: How many documents can I upload?**  
A: No strict limit, but 5-10 focused documents typically provide the best results.

**Q: Can I add documents after creation?**  
A: You would need to create a new GPT with the updated document set.

### Performance Questions

**Q: Which model should I choose?**  
A: Start with GPT-4o-mini for most use cases. Upgrade to larger models for complex reasoning tasks.

**Q: What temperature setting is best?**  
A: 0.7 provides good balance. Lower (0.3-0.5) for factual tasks, higher (1.0-1.5) for creative tasks.

**Q: Will my GPT remember previous conversations?**  
A: Within a session, yes. Context is maintained during the chat conversation.

### Troubleshooting

**Q: My GPT's responses don't match my expectations. What should I do?**  
A: Try adjusting the temperature, or create a new version with more specific instructions in Step 6.

**Q: Documents aren't being referenced.**  
A: Ensure documents were successfully processed (green checkmarks). Verify questions relate to document content.

**Q: How do I switch back to regular chat?**  
A: Navigate to the Chat tab. Custom GPT mode only activates when you start from the Custom GPT creator.

## Getting Started Checklist

Before creating your Custom GPT, prepare:
- [ ] Clear idea of the problem you want to solve
- [ ] Understanding of who will use it
- [ ] Examples of ideal interactions
- [ ] Reference documents (if applicable)
- [ ] Preferred communication style

During creation:
- [ ] Answer all guided questions thoughtfully
- [ ] Review generated system prompt
- [ ] Configure appropriate settings
- [ ] Test with representative queries

After activation:
- [ ] Try various types of questions
- [ ] Verify boundaries are respected
- [ ] Check document citations (if applicable)
- [ ] Adjust temperature if needed

## Using Custom GPT APIs from Other Frontends

### Integration Overview

The Custom GPT functionality is accessible via REST APIs, making it possible to integrate into any frontend framework (React, Vue, Angular, mobile apps, etc.).

### Quick Start for Developers

1. **Review API Documentation**: See [Custom GPT API Integration Guide](./CUSTOM_GPT_API_INTEGRATION.md) for complete API reference
2. **Authentication Setup**: Obtain and configure authentication tokens
3. **Implement Workflow**: Follow the 9-step workflow outlined in the API guide
4. **Handle Errors**: Implement proper error handling for all API calls

### Key Integration Points

**Initialization**:
```javascript
// Detect language and get first question
const { language } = await detectLanguage(initialIdea);
const question = await getPurposeQuestion(initialIdea, language);
```

**Configuration Steps**:
```javascript
// For each step (1-6):
// 1. Display question and examples
// 2. Get user input
// 3. Resolve answer if needed
const resolvedAnswer = await resolveAnswer(userInput, question, examples);
// 4. Get next question
const nextQuestion = await getNextQuestion(stepNumber, resolvedAnswer);
```

**System Prompt & Name Generation**:
```javascript
const { gpt_name, system_prompt } = await generateSystemPrompt({
  name, purpose, audience, tone, capabilities,
  constraints, knowledge, example_interaction,
  custom_instructions, language
});
// gpt_name is auto-generated (2-5 words, in detected language)
// User can edit it in settings if desired
```

**Document Upload**:
```javascript
const { file_ids, session_id } = await uploadDocuments(files, username);
// Important: Set custom_gpt: true in upload request
```

**Chat Initialization**:
```javascript
// Determine endpoint based on documents
const endpoint = fileIds.length > 0 ? '/file/chat' : '/chat/anthropic';

const response = await sendChatMessage(message, {
  endpoint,
  system_prompt,
  file_ids,
  session_id,
  model_choice,
  temperature
});
```

### Example Frontend Implementations

**React/Next.js**: See JavaScript examples in [API Integration Guide](./CUSTOM_GPT_API_INTEGRATION.md#example-implementations)

**Python/Flask**: See Python examples in [API Integration Guide](./CUSTOM_GPT_API_INTEGRATION.md#example-implementations)

**Mobile Apps**: Use standard HTTP client libraries (fetch, axios, etc.) with authentication headers

### Required API Endpoints

All endpoints require authentication (Bearer token):

1. `POST /language/detect` - Language detection
2. `POST /chat/azure` - Question generation & answer resolution
3. `POST /gpt/generate-system-prompt` - System prompt generation
4. `POST /file/upload` - Document upload (with `custom_gpt: true`)
5. `GET /available-models` - Get available models
6. `POST /file/chat` - Chat with documents
7. `POST /chat/anthropic` - Chat without documents

### Authentication

All API endpoints require Bearer token authentication:

```javascript
headers: {
  'Authorization': `Bearer ${authToken}`,
  'Content-Type': 'application/json'
}
```

For file uploads, authentication is still required but Content-Type is set automatically by the browser.

### Error Handling

Implement proper error handling for:
- **403 Forbidden**: Authentication issues - verify token
- **400 Bad Request**: Missing or invalid parameters
- **500 Server Error**: Retry logic recommended
- **Network Errors**: Handle timeouts and connection issues

See [Error Handling section](./CUSTOM_GPT_API_INTEGRATION.md#error-handling) in API guide for details.

## Support and Resources

### Need Help?
- Review the guided examples provided at each step
- Start with simpler GPTs before attempting complex configurations
- Test incrementally and refine based on results
- Check [API Integration Guide](./CUSTOM_GPT_API_INTEGRATION.md) for technical details

### Learn More
- Explore example use cases above for inspiration
- Experiment with different temperature settings
- Try various communication styles to find what works
- Review [API Integration Guide](./CUSTOM_GPT_API_INTEGRATION.md) for complete API reference

### Documentation
- **[Custom GPT API Integration Guide](./CUSTOM_GPT_API_INTEGRATION.md)** - Complete API reference for developers
- **[Streamlit Implementation](../streamlit_components/custom_gpt_creator.py)** - Reference implementation
- **[Main App Integration](../streamlit_app.py)** - Chat integration example

## Conclusion

Custom GPT puts powerful AI customization at your fingertips without requiring technical expertise. Whether you're building a specialized assistant for your business, an educational tool for students, or a creative partner for content generation, the guided process ensures you create an effective, purpose-built AI assistant.

Start creating your Custom GPT today and experience the power of personalized AI assistance tailored to your exact needs.
