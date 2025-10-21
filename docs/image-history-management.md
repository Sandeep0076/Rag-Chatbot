# Image History Management - Complete Guide

## Overview

Context-aware image generation feature that enables follow-up modifications to previously generated images by maintaining conversation history and using intelligent LLM-based prompt rewriting. The system automatically detects whether a user wants to modify a previous image or create a completely new one.

## ✅ **Key Features**

- **Smart Context Detection**: LLM intelligently determines modification vs new request
- **LLM-Based Rewriting**: Uses Azure OpenAI GPT-4o-mini for intelligent prompt merging
- **Session-Based History**: In-memory prompt history per user session
- **Memory Efficient**: Simple array storage in session state
- **Backward Compatible**: `prompt_history` field is optional
- **Model Agnostic**: Works with DALL-E 3 & Imagen
- **Transparency**: Response includes `final_prompt`, `used_context`, `rewrite_method`, `context_type`

## Architecture

### Components

#### 1. **Prompt History Management**
- **In-Memory Storage**: Session-based prompt history (Streamlit: `st.session_state.image_prompt_history`)
- **No Persistent Storage**: History is maintained in session state, not saved to disk (because in frontend its saved in database and from there it would be called)
- **Format**: Simple array of strings: `["prompt1", "prompt2", "prompt3"]`
- **Management**:
  - Appends final prompts after successful generation
  - History persists only during active session (streamlit)

#### 2. **ImagePromptRewriter** (`rtl_rag_chatbot_api/chatbot/utils/image_prompt_rewriter.py`)
- **LLM-based rewriting only** with intelligent decision-making
- Uses Azure OpenAI GPT-4o-mini for prompt merging
- **Smart Detection**: Determines if request is modification or new request
- **Content-Filter Safe**: Uses collaborative language to avoid Azure content policy violations

#### 3. **Enhanced Request Schema** (`rtl_rag_chatbot_api/common/models.py`)
```python
class ImageGenerationRequest(BaseModel):
    prompt: List[str]  # Array of prompts (history + current), similar to chat endpoint
    size: str = "1024x1024"
    n: int = 1
    model_choice: Optional[str] = None  # "dall-e-3" or "imagen-3.0"
```

**Consistent with Chat Endpoint:**
```python
class Query(BaseModel):
    text: List[str]  # Chat uses 'text' array
    # ... other fields

class ImageGenerationRequest(BaseModel):
    prompt: List[str]  # Image uses 'prompt' array
    # ... other fields
```

#### 4. **Updated Endpoint** (`/image/generate` in `app.py`)
- Accepts `prompt` array (similar to chat `text` array)
- Extracts history (all but last) and current prompt (last element)
- Rewrites prompts using LLM with smart decision-making
- Returns metadata: `final_prompt`, `used_context`, `rewrite_method`, `context_type`
- Session management handled by Streamlit UI

## Smart Context Detection

### How It Works

The LLM analyzes user requests and makes intelligent decisions:

1. **Analyze the new instruction** to determine intent:
   - **MODIFICATION**: "make it blue", "add a dog", "change the background", "remove the hat"
   - **NEW REQUEST**: "create a sunset landscape", "draw a robot", "make a logo", "generate a portrait"

2. **Apply appropriate logic**:
   - **If MODIFICATION**: Preserve original scene/style + apply only requested change
   - **If NEW REQUEST**: Use new instruction as-is, ignore original prompt

### Example 1: Modification
```json
// Request with history:
{
  "prompt": ["a red sports car in a futuristic city", "make it blue"]
}
```

**API Processing:**
- History: `["a red sports car in a futuristic city"]`
- Current: `"make it blue"`
- **LLM Analysis**: "make it blue" → MODIFICATION  
- **Result**: "a blue sports car in a futuristic city"  
- **Response**: `"context_type": "modification"`

### Example 2: New Request
```json
// Request with history but new topic:
{
  "prompt": ["a red sports car in a futuristic city", "create a peaceful forest scene with birds"]
}
```

**API Processing:**
- History: `["a red sports car in a futuristic city"]`
- Current: `"create a peaceful forest scene with birds"`
- **LLM Analysis**: "create a peaceful forest scene" → NEW REQUEST  
- **Result**: "create a peaceful forest scene with birds"  
- **Response**: `"context_type": "new_request"`

## Workflow

### First Request (Cold Start)
```json
POST /image/generate
{
  "prompt": ["A red sports car in a futuristic city at night with neon lights, cyberpunk style, 4k"],
  "size": "1024x1024",
  "model_choice": "dall-e-3"
}
```

**Response:**
```json
{
  "success": true,
  "image_urls": ["data:image/png;base64,..."],
  "final_prompt": "A red sports car in a futuristic city at night with neon lights, cyberpunk style, 4k",
  "used_context": false,
  "rewrite_method": "none",
  "context_type": "none",
  "model": "dall-e-3",
  "size": "1024x1024"
}
```

### Follow-up Request (Context-Aware)
```json
POST /image/generate
{
  "prompt": [
    "A red sports car in a futuristic city at night with neon lights, cyberpunk style, 4k",
    "make the car blue"
  ],
  "model_choice": "dall-e-3"
}
```

**Internal Process:**
1. API extracts: `history = prompt[:-1]`, `current = prompt[-1]`
2. Sends to LLM with smart decision-making prompt
3. LLM analyzes: "make the car blue" → MODIFICATION
4. LLM generates: "A blue sports car in a futuristic city at night with neon lights, cyberpunk style, 4k"
5. Generates image with rewritten prompt
6. Returns result with metadata (Streamlit appends final_prompt to session history)

**Response:**
```json
{
  "success": true,
  "image_urls": ["data:image/png;base64,..."],
  "final_prompt": "A blue sports car in a futuristic city at night with neon lights, cyberpunk style, 4k",
  "used_context": true,
  "rewrite_method": "llm",
  "context_type": "modification",
  "model": "dall-e-3",
  "size": "1024x1024"
}
```

## LLM Prompt Engineering

### Current Implementation (Content-Filter Safe)
```
You are helping refine an image generation prompt.

CONTEXT - Previous image was generated with this prompt:
"{base_prompt}"

USER'S NEW REQUEST:
"{instruction}"

YOUR TASK:
Step 1: Analyze the user's request and decide:
  • Is this a MODIFICATION of the previous image?
    (Examples: 'make it blue', 'add a tree', 'change background to sunset', 'remove the hat')
  • OR is this a COMPLETELY NEW image request?
    (Examples: 'create a forest scene', 'draw a robot', 'generate a logo', 'make a portrait')

Step 2: Based on your decision:
  • If MODIFICATION: Merge the previous prompt with the new request.
    Preserve all original details (scene, objects, style, mood, lighting)
    and apply only the specific change requested.
  • If NEW REQUEST: Use the new request as-is.
    The previous prompt is not relevant.

IMPORTANT:
  • For modifications, create a complete standalone prompt with all details explicitly stated
  • For new requests, output the new request directly
  • Keep prompts clear and detailed
  • Output ONLY the final prompt text, no explanations

FINAL PROMPT:
```

### Context Type Detection
Simple heuristic to determine how LLM interpreted the request:
- **Word overlap analysis**: If rewritten prompt is very similar to current prompt → likely new request
- **Threshold**: >70% word overlap suggests new request, otherwise modification

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Whether generation succeeded |
| `image_urls` | list | Generated image URLs (data URLs for Imagen, HTTP for DALL-E) |
| `final_prompt` | str | Actual prompt used (after rewriting) |
| `used_context` | bool | Whether historical context was applied |
| `rewrite_method` | str | Method used: "llm", "none", or "error" |
| `context_type` | str | **NEW**: "modification", "new_request", or "none" |
| `model` | str | Model used for generation |
| `size` | str | Image size |
| `prompt` | str | Original user prompt (for reference) |

### Context Type Values
- `"modification"`: LLM treated as modification to previous image
- `"new_request"`: LLM treated as completely new image request  
- `"none"`: No context was used

## Context Detection Logic

Context is used when:
- `prompt` array has more than one element (length > 1)
- API automatically extracts history (`prompt[:-1]`) and current (`prompt[-1]`)
- Similar to chat endpoint: `text` array → last element is current question

## Error Handling

- **No LLM available**: Returns original instruction, `rewrite_method: "none"`
- **LLM rewrite fails**: Returns original instruction, `rewrite_method: "error"`
- **No historical context**: Treats as new request, `used_context: false`
- **Empty prompt array**: Returns 400 error "Prompt array cannot be empty"
- **Single element array**: No history, uses prompt as-is
- **Azure content filter**: Uses simplified prompt to avoid jailbreak detection

## Testing Examples

### Example 1: Color Change
```bash
# First image
curl -X POST http://localhost:8080/image/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": ["A superhero in a blue costume flying over a city"],
    "model_choice": "dall-e-3"
  }'

# Follow-up (client manages history by building array)
curl -X POST http://localhost:8080/image/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": ["A superhero in a blue costume flying over a city", "change costume to red"],
    "model_choice": "dall-e-3"
  }'
```

### Example 2: Multiple Sequential Changes
```bash
# First image
curl -X POST http://localhost:8080/image/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": ["A peaceful forest scene with morning sunlight, birds flying"]
  }'

# Follow-up 1
curl -X POST http://localhost:8080/image/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": [
      "A peaceful forest scene with morning sunlight, birds flying",
      "make it sunset instead of morning"
    ]
  }'

# Follow-up 2 (history builds up in array)
curl -X POST http://localhost:8080/image/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": [
      "A peaceful forest scene with morning sunlight, birds flying",
      "A peaceful forest scene with sunset lighting, birds flying",
      "add a deer in the clearing"
    ]
  }'
```

### Example 3: New Request Detection
```bash
# With history but requesting completely new image
curl -X POST http://localhost:8080/image/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": ["a cat sitting on a chair", "draw a robot in space"]
  }'
# Response: {"context_type": "new_request", "final_prompt": "draw a robot in space"}
```

## Files Modified/Created

### New Files
- `rtl_rag_chatbot_api/chatbot/utils/image_prompt_rewriter.py`

### Modified Files
- `rtl_rag_chatbot_api/common/models.py` - Changed `prompt` from `str` to `List[str]` (consistent with chat endpoint)
- `rtl_rag_chatbot_api/app.py` - Updated `/image/generate` and `/image/generate-combined` endpoints with array handling
- `streamlit_image_generation.py` - Updated to build and send prompt array (similar to chat)

## Performance Considerations

- **LLM Latency**: ~1-2 seconds per rewrite (GPT-4o-mini)
- **Storage**: In-memory only (session state), no disk I/O
- **Memory**: Simple array storage, minimal overhead
- **Concurrency**: Session-isolated, no cross-session interference
- **Content Filter**: Simplified prompts reduce Azure policy violations

## Streamlit UI Enhancements

### Smart Context Indicators
- **Modification**: "✅ Modified previous image (preserved original style)"
- **New Request**: "✅ Created new image (ignored previous context)"
- **Fallback**: "✅ Used context from previous prompts (method: llm)"

### User Experience
Users get clear feedback about how their request was interpreted:
- **Modification requests** → Shows preservation of original style
- **New requests** → Shows that previous context was ignored
- **Transparency** → Always shows the final prompt used
- **History View**: Expandable section showing all previous prompts
- **Clear History**: Button to reset prompt history in session

### Session State Management (Streamlit)
```python
# In streamlit_image_generation.py
st.session_state.image_prompt_history = []  # Initialize

# After successful generation
final_prompt = result.get("final_prompt", prompt)
st.session_state.image_prompt_history.append(final_prompt)

# When making request - build prompt array
if st.session_state.image_prompt_history:
    prompt_array = st.session_state.image_prompt_history + [current_prompt]
else:
    prompt_array = [current_prompt]

# Send to API
generate_image(..., prompt_array)  # API receives: {"prompt": ["p1", "p2", "current"]}
```

## Future Enhancements (Optional)

1. **Persistent History**: Save history to database for cross-session continuity
2. **Auto-captioning**: Generate short captions for better recall
3. **Style extraction**: Parse prompts to extract style descriptors
4. **History Branching**: Allow users to branch from any previous prompt
5. **Export History**: Download prompt history as JSON
6. **Rule-based tier**: Fast heuristics for simple edits (color, style changes)

## Maintenance

- **History Storage**: In-memory only (Streamlit session state)
- **Clear History**: Click "Clear History" button in UI
- **Session Lifetime**: History persists only during active browser session
- **No Cleanup Required**: Memory automatically released when session ends

## Troubleshooting

### Common Issues

1. **Azure Content Filter Violations**
   - **Symptom**: "jailbreak detected" errors
   - **Solution**: Use simplified, collaborative language in prompts
   - **Status**: ✅ Fixed with current implementation

2. **Context Not Applied**
   - **Check**: `prompt` array has more than one element
   - **Check**: Client is building array correctly (history + current)
   - **Check**: Streamlit session state has history

3. **Wrong Context Type Detection**
   - **Check**: LLM prompt clarity
   - **Check**: Word overlap threshold (70%)
   - **Debug**: Check `context_type` in API response

4. **History Lost Between Requests**
   - **Cause**: Streamlit session reset or page refresh
   - **Solution**: History is session-based, expected behavior
   - **Future**: Consider persistent storage for cross-session continuity

## Summary

The image history management system provides intelligent, context-aware image generation with:

✅ **Smart Detection**: Automatically determines modification vs new request  
✅ **Content-Filter Safe**: Avoids Azure policy violations  
✅ **Session-Based**: Simple in-memory history management  
✅ **Memory Efficient**: Minimal overhead, no disk I/O  
✅ **Transparent**: Clear feedback on how requests are interpreted  
✅ **Backward Compatible**: `prompt_history` is optional  
✅ **Client-Managed**: API is stateless, clients control history  

The system makes iterative image generation much more intuitive and user-friendly! 🎯
