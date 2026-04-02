# Bathroom Access Concierge

Bathroom Access Concierge is a conversational AI agent for finding public restrooms with accessibility-aware filtering, follow-up refinement, remembered preferences, and route-based suggestions.

It is built as an AI layer on top of restroom data, with a chat interface that lets users ask natural questions like:

- “Show me accessible restrooms near Back Bay.”
- “I’m at Fisherman’s Wharf and heading to Union Square. Suggest one along the way.”
- “What about one closer to the destination?”
- “Remember that I prefer single-stall restrooms.”

## Live demo

Deployed app:

`https://cf-ai-amithsaligrama.saligrama-amith.workers.dev/`

## What the app does

Bathroom Access Concierge lets users:

- search for restrooms by city, neighborhood, landmark, or route
- filter for wheelchair accessibility, single-stall preference, changing tables, and open-now needs
- ask follow-up questions without restarting the search
- store simple user preferences across the conversation
- get clearer restroom recommendations instead of raw map/database rows

## Best cities to try first

**For the strongest demo experience, ask about Boston, New York City, or San Francisco.**

Those cities currently have the most complete and best-tested restroom coverage in this project, so they produce the most reliable location filtering, accessibility matching, and route-based suggestions.

Recommended city phrasing:

- **Boston**: Back Bay, Seaport, North End, South Station, Charlestown
- **New York City**: Manhattan, Times Square, Midtown, SoHo, Lower Manhattan
- **San Francisco**: Fisherman’s Wharf, Union Square, SoMa, Mission District, Financial District

If you ask about other cities, the app may still return results, but coverage is less consistent.

## How to use the application

The app works best when the user gives it location context first, then adds constraints.

### 1. Start with a location

Use a city, neighborhood, landmark, or route.

Examples:

- `Show me restroom options near Back Bay in Boston.`
- `Find a restroom near Times Square in New York City.`
- `Show me options near Fisherman’s Wharf in San Francisco.`

### 2. Add what matters to you

The app can use constraints such as:

- wheelchair accessible
- single-stall
- open now
- changing table

Examples:

- `Find a wheelchair-accessible restroom near Back Bay.`
- `Show me open restrooms with a changing table near Union Square.`
- `I prefer single-stall restrooms.`

### 3. Use follow-up questions

Once the app has a search context, you can refine without repeating everything.

Examples:

- `What about one closer to the destination?`
- `Can you keep it in the same city?`
- `Show me another option.`
- `Only show accessible ones.`

### 4. Ask route-based questions

You can search along a route by giving both a start and end point.

Examples:

- `I’m at Back Bay and going to Charlestown. Show accessible bathrooms along the way.`
- `I’m near Fisherman’s Wharf and heading to Union Square. Suggest one on the way.`

### 5. Save preferences

The agent remembers basic preferences during the session.

Examples:

- `Remember that I need wheelchair access.`
- `Remember that I prefer single-stall restrooms.`
- `I usually care about changing tables.`

### 6. Reset when switching cities

If you radically change context, use **Clear** in the UI before starting a new city search. This avoids carrying over route or preference state from the previous conversation.

## Suggested demo prompts

If you want a fast demo flow, use these:

### Boston

- `Show me accessible restrooms near Back Bay.`
- `I’m at Back Bay and going to Charlestown. Show accessible bathrooms along the way.`
- `What about one closer to the destination?`

### New York City

- `Find a restroom near Times Square in New York City.`
- `Only show wheelchair-accessible options.`
- `Show me one closer to SoHo.`

### San Francisco

- `Show me restroom options near Fisherman’s Wharf in San Francisco.`
- `I’m heading to Union Square. Suggest one along the way.`
- `Remember that I prefer single-stall restrooms.`

## Architecture

This project is designed to satisfy the application requirements for an AI-powered app with:

- an LLM
- workflow / coordination
- user input through chat
- memory / state

### Components

- **Frontend:** React + Vite
- **Backend:** Cloudflare Workers + Agents
- **Agent runtime:** `AIChatAgent`
- **LLM:** Workers AI
- **State:** persistent agent state for user preferences and current search context
- **Data layer:** restroom dataset exported into app-readable TypeScript data

### Workflow / coordination

The app’s search flow is:

1. parse the user request
2. detect explicit city / area / route context
3. merge request filters with remembered preferences
4. filter candidate restrooms by city and request constraints
5. score and rank results
6. return natural-language recommendations

## Running locally

Install dependencies:

```bash
npm install
```

Run the dev server:

```bash
npm run dev
```

Open the local URL shown in the terminal, typically:

```text
http://localhost:5173
```

## Deploying

Deploy to Cloudflare Workers:

```bash
npm run deploy
```

## Files of interest

- `src/server.ts` — main agent logic, filtering, ranking, and state handling
- `src/app.tsx` — chat UI
- `src/data/restrooms.ts` — restroom dataset used by the app
- `PROMPTS.md` — prompts used during AI-assisted development

## Current limitations

- Coverage is strongest in **Boston, New York City, and San Francisco**.
- Results outside those cities may be sparse or less well-tested.
- Accessibility details depend on the quality of the underlying source data.
- Route suggestions are strongest when the app has good coordinates for both the user’s intended area and restroom candidates.

## Why this project exists

Bathroom Access Concierge extends the broader Bathroom Access mission with a conversational AI interface. Instead of making users sift through raw records, it helps them find suitable restroom options through natural language, remembered preferences, and iterative refinement.

## AI development notes

This repository should also include a `PROMPTS.md` file documenting prompts used during AI-assisted development.
