import { createWorkersAI } from "workers-ai-provider";
import { routeAgentRequest, callable } from "agents";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  streamText,
  convertToModelMessages,
  pruneMessages,
  tool,
  stepCountIs,
  createUIMessageStream,
  createUIMessageStreamResponse
} from "ai";
import { z } from "zod";
import { RESTROOMS, type RestroomEntry } from "./data/restrooms";

type Restroom = RestroomEntry;

type UserPreferences = {
  wheelchairAccessible: boolean;
  singleStall: boolean;
  openNow: boolean;
  changingTable: boolean;
};

type RestroomSearchFilters = {
  areaQuery?: string;
  fromArea?: string;
  toArea?: string;
  wheelchairAccessible?: boolean;
  singleStall?: boolean;
  openNow?: boolean;
  changingTable?: boolean;
};

type ConciergeState = {
  preferences: UserPreferences;
  lastLocation?: {
    areaQuery: string;
  };
};

type OSMSearchResult = {
  display_name: string;
  lat: string;
  lon: string;
  osm_id: number;
  osm_type: "node" | "way" | "relation";
};

const DEFAULT_PREFERENCES: UserPreferences = {
  wheelchairAccessible: false,
  singleStall: false,
  openNow: false,
  changingTable: false
};

function mergePreferences(
  current: UserPreferences,
  update: Partial<UserPreferences>
): UserPreferences {
  return {
    wheelchairAccessible:
      update.wheelchairAccessible ?? current.wheelchairAccessible,
    singleStall: update.singleStall ?? current.singleStall,
    openNow: update.openNow ?? current.openNow,
    changingTable: update.changingTable ?? current.changingTable
  };
}

function formatRestroom(restroom: Restroom, reasons: string[]) {
  const openNow = isLikelyOpenNow(restroom);
  const features = {
    wheelchairAccessible: isEffectivelyWheelchairAccessible(restroom),
    singleStall: isEffectivelySingleStall(restroom),
    openNow,
    changingTable: isEffectivelyChangingTable(restroom)
  };
  return {
    id: restroom.id,
    name: restroom.name,
    address: restroom.address,
    city: restroom.city,
    features,
    notes: restroom.notes,
    recommendationReasons: reasons
  };
}

function pickRequestedFeatures(
  allFeatures: {
    wheelchairAccessible: boolean;
    singleStall: boolean;
    openNow: boolean;
    changingTable: boolean;
  },
  requested: {
    wheelchairAccessible: boolean;
    singleStall: boolean;
    openNow: boolean;
    changingTable: boolean;
  }
) {
  const result: Record<string, boolean> = {};
  if (requested.wheelchairAccessible) {
    result.wheelchairAccessible = allFeatures.wheelchairAccessible;
  }
  if (requested.singleStall) {
    result.singleStall = allFeatures.singleStall;
  }
  if (requested.openNow) {
    result.openNow = allFeatures.openNow;
  }
  if (requested.changingTable) {
    result.changingTable = allFeatures.changingTable;
  }
  return result;
}

function getNoteFieldValue(
  notes: string,
  label: "Hours" | "Remarks"
): string | null {
  const match = notes.match(new RegExp(`${label}:\\s*([^|]+)`));
  return match?.[1]?.trim() ?? null;
}

function getExplicitFeatureValue(
  restroom: Restroom,
  feature: "wheelchairAccessible" | "singleStall" | "changingTable" | "openNow"
): boolean | null {
  const text =
    `${restroom.name} ${restroom.address} ${restroom.city} ${restroom.notes}`.toLowerCase();

  if (feature === "wheelchairAccessible") {
    if (
      text.includes("not wheelchair accessible") ||
      text.includes("no wheelchair access") ||
      text.includes("inaccessible")
    ) {
      return false;
    }
    if (
      text.includes("wheelchair accessible") ||
      text.includes("ada") ||
      text.includes("accessible restroom")
    ) {
      return true;
    }
    return null;
  }

  if (feature === "singleStall") {
    if (
      text.includes("multi stall") ||
      text.includes("multiple stalls") ||
      text.includes("not single stall")
    ) {
      return false;
    }
    if (
      text.includes("single stall") ||
      text.includes("single-stall") ||
      text.includes("unisex stall")
    ) {
      return true;
    }
    return null;
  }

  if (feature === "changingTable") {
    if (
      text.includes("no changing table") ||
      text.includes("without changing table") ||
      text.includes("no baby changing")
    ) {
      return false;
    }
    if (
      text.includes("changing table") ||
      text.includes("baby changing") ||
      text.includes("diaper station")
    ) {
      return true;
    }
    return null;
  }

  // openNow explicit from DB notes only when clearly encoded.
  if (
    text.includes("temporarily closed") ||
    text.includes("currently closed")
  ) {
    return false;
  }
  if (
    text.includes("24/7") ||
    text.includes("24 hours") ||
    text.includes("24hr") ||
    text.includes("24 hrs")
  ) {
    return true;
  }
  return null;
}

function getDatabaseDetails(restroom: Restroom) {
  return {
    hours: getNoteFieldValue(restroom.notes, "Hours"),
    remarks: getNoteFieldValue(restroom.notes, "Remarks"),
    features: {
      wheelchairAccessible: getExplicitFeatureValue(
        restroom,
        "wheelchairAccessible"
      ),
      singleStall: getExplicitFeatureValue(restroom, "singleStall"),
      changingTable: getExplicitFeatureValue(restroom, "changingTable"),
      openNow: getExplicitFeatureValue(restroom, "openNow")
    }
  };
}

function hasAccessibilityExclusionSignal(restroom: Restroom): boolean {
  const text =
    `${restroom.name} ${restroom.address} ${restroom.city} ${restroom.notes}`.toLowerCase();
  const facilityIsEmergencyService =
    text.includes("firehouse") ||
    text.includes("fire station") ||
    text.includes("engine ") ||
    text.includes("police") ||
    text.includes("precinct") ||
    text.includes("sheriff");
  return (
    facilityIsEmergencyService ||
    text.includes("inaccessible when firefighters are out") ||
    text.includes("bathrooms are inaccessible") ||
    text.includes("not wheelchair accessible") ||
    text.includes("no wheelchair access") ||
    text.includes("no accessible restroom") ||
    text.includes("inaccessible")
  );
}

function isEffectivelyWheelchairAccessible(restroom: Restroom): boolean {
  // Product rule: assume wheelchair accessibility unless explicitly excluded.
  return !hasAccessibilityExclusionSignal(restroom);
}

function hasSingleStallExclusionSignal(restroom: Restroom): boolean {
  const text =
    `${restroom.name} ${restroom.address} ${restroom.city} ${restroom.notes}`.toLowerCase();
  return (
    text.includes("multi stall") ||
    text.includes("multiple stalls") ||
    text.includes("multi-stall") ||
    text.includes("not single stall")
  );
}

function isEffectivelySingleStall(restroom: Restroom): boolean {
  // Product rule: assume single-stall availability unless explicitly excluded.
  return !hasSingleStallExclusionSignal(restroom);
}

function hasChangingTableExclusionSignal(restroom: Restroom): boolean {
  const text =
    `${restroom.name} ${restroom.address} ${restroom.city} ${restroom.notes}`.toLowerCase();
  return (
    text.includes("no changing table") ||
    text.includes("without changing table") ||
    text.includes("no baby changing") ||
    text.includes("no diaper station")
  );
}

function isEffectivelyChangingTable(restroom: Restroom): boolean {
  // Product rule: assume changing table availability unless explicitly excluded.
  return !hasChangingTableExclusionSignal(restroom);
}

function getTimezoneForRestroom(restroom: Restroom): string {
  const haystack =
    `${restroom.address} ${restroom.city} ${restroom.notes}`.toLowerCase();
  if (
    haystack.includes(", ca") ||
    haystack.includes(" california") ||
    haystack.includes("san francisco") ||
    haystack.includes("fisherman's wharf") ||
    haystack.includes("941")
  ) {
    return "America/Los_Angeles";
  }
  return "America/New_York";
}

function getMinutesNowInTimezone(timeZone: string): number {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23"
  }).formatToParts(new Date());

  const hour = Number(parts.find((p) => p.type === "hour")?.value ?? "0");
  const minute = Number(parts.find((p) => p.type === "minute")?.value ?? "0");
  return hour * 60 + minute;
}

function parseOpenWindow(notes: string): { start: number; end: number } | null {
  const source = notes.toLowerCase();

  // 24-hour style, e.g. 07:00-22:00
  const m24 = source.match(/\b(\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})\b/);
  if (m24) {
    const start = Number(m24[1]) * 60 + Number(m24[2]);
    const end = Number(m24[3]) * 60 + Number(m24[4]);
    return { start, end };
  }

  // 12-hour style, e.g. 6:30 AM - 9 PM
  const m12 = source.match(
    /\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m?\.?\s*[-–]\s*(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m?\.?/i
  );
  if (!m12) return null;

  const toMinutes = (
    hourRaw: string,
    minuteRaw: string | undefined,
    suffix: string
  ) => {
    let hour = Number(hourRaw) % 12;
    if (suffix.toLowerCase() === "p") hour += 12;
    const minute = Number(minuteRaw ?? "0");
    return hour * 60 + minute;
  };

  return {
    start: toMinutes(m12[1], m12[2], m12[3]),
    end: toMinutes(m12[4], m12[5], m12[6])
  };
}

function isLikelyOpenNow(restroom: Restroom): boolean {
  const notes = restroom.notes.toLowerCase();
  if (
    notes.includes("24/7") ||
    notes.includes("24 hours") ||
    notes.includes("24hr") ||
    notes.includes("24 hrs")
  ) {
    return true;
  }

  if (
    notes.includes("temporarily closed") ||
    notes.includes("currently closed") ||
    notes.includes("closed for renovation")
  ) {
    return false;
  }

  const window = parseOpenWindow(restroom.notes);
  if (window) {
    const now = getMinutesNowInTimezone(getTimezoneForRestroom(restroom));
    // Handle overnight windows like 22:00-02:00
    if (window.start <= window.end) {
      return now >= window.start && now <= window.end;
    }
    return now >= window.start || now <= window.end;
  }

  // Product rule: assume open now unless explicitly closed.
  return true;
}

function isMatchingRequirements(
  restroom: Restroom,
  requirements: UserPreferences
): boolean {
  return (
    (!requirements.wheelchairAccessible ||
      isEffectivelyWheelchairAccessible(restroom)) &&
    (!requirements.singleStall || isEffectivelySingleStall(restroom)) &&
    (!requirements.openNow || isLikelyOpenNow(restroom)) &&
    (!requirements.changingTable || isEffectivelyChangingTable(restroom))
  );
}

function scoreRestroom(
  restroom: Restroom,
  requirements: UserPreferences
): {
  score: number;
  reasons: string[];
} {
  let score = 0;
  const reasons: string[] = [];

  if (
    requirements.wheelchairAccessible &&
    isEffectivelyWheelchairAccessible(restroom)
  ) {
    score += 3;
    reasons.push("wheelchair accessible");
  }
  if (hasAccessibilityExclusionSignal(restroom)) {
    score -= 6;
  }
  if (requirements.singleStall && isEffectivelySingleStall(restroom)) {
    score += 3;
    reasons.push("single stall");
  }
  if (requirements.openNow && isLikelyOpenNow(restroom)) {
    score += 2;
    reasons.push("likely open now");
  }
  if (requirements.changingTable && isEffectivelyChangingTable(restroom)) {
    score += 2;
    reasons.push("has changing table");
  }

  if (reasons.length === 0) {
    reasons.push("matches general request");
  }

  return { score, reasons };
}

function tokenizeAreaQuery(query: string): string[] {
  const stopWords = new Set([
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "give",
    "go",
    "i",
    "im",
    "is",
    "it",
    "its",
    "looking",
    "me",
    "near",
    "nearby",
    "my",
    "not",
    "of",
    "on",
    "options",
    "or",
    "please",
    "restroom",
    "restrooms",
    "bathroom",
    "bathrooms",
    "route",
    "find",
    "closest",
    "need",
    "wheelchair",
    "accessible",
    "single",
    "stall",
    "open",
    "now",
    "changing",
    "table",
    "show",
    "suggest",
    "suggestion",
    "that",
    "this",
    "to",
    "way",
    "where",
    "downtown",
    "uptown",
    "midtown",
    "station",
    "area",
    "the",
    "in",
    "at",
    "around"
  ]);

  return query
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 1 && !stopWords.has(token));
}

function extractLatestUserText(messages: unknown): string | undefined {
  if (!Array.isArray(messages)) return undefined;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i] as {
      role?: string;
      content?: string;
      parts?: Array<{ type?: string; text?: string }>;
    };
    if (msg.role !== "user") continue;
    if (typeof msg.content === "string" && msg.content.trim()) {
      return msg.content.trim();
    }
    if (Array.isArray(msg.parts)) {
      const text = msg.parts
        .filter((p) => p?.type === "text" && typeof p.text === "string")
        .map((p) => p.text?.trim() || "")
        .filter(Boolean)
        .join(" ")
        .trim();
      if (text) return text;
    }
  }
  return undefined;
}

function userExplicitlyAskedForFeatureLookup(userText?: string): boolean {
  if (!userText) return false;
  const text = userText.toLowerCase();
  const triggers = [
    "hours",
    "open now",
    "opening hours",
    "wheelchair",
    "accessible",
    "single stall",
    "changing table",
    "verify",
    "check details",
    "look up",
    "lookup",
    "confirm features"
  ];
  return triggers.some((phrase) => text.includes(phrase));
}

function isLikelyFollowUpFeatureQuery(text?: string): boolean {
  if (!text) return false;
  const t = text.toLowerCase();
  return (
    t.includes("open now") ||
    t.includes("currently open") ||
    t.includes("hours") ||
    t.includes("wheelchair") ||
    t.includes("single stall") ||
    t.includes("changing table")
  );
}

function computeAreaRelevance(restroom: Restroom, areaQuery?: string): number {
  if (!areaQuery) return 0;
  const haystack =
    `${restroom.name} ${restroom.address} ${restroom.city}`.toLowerCase();
  const fullQuery = areaQuery.toLowerCase().trim();
  const tokens = tokenizeAreaQuery(areaQuery).filter((t) => t.length >= 3);
  let score = 0;
  if (fullQuery && haystack.includes(fullQuery)) score += 6;
  for (const token of tokens) {
    if (haystack.includes(token)) score += 2;
    if (normalizeLocation(restroom.city) === token) score += 3;
  }
  return score;
}

function normalizeLocation(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .trim();
}

type RouteProfile = {
  include: string[];
  exclude: string[];
};

function findRouteProfile(fromArea?: string, toArea?: string): RouteProfile {
  if (!fromArea || !toArea) return { include: [], exclude: [] };
  const from = normalizeLocation(fromArea);
  const to = normalizeLocation(toArea);
  const dynamic = [
    ...tokenizeAreaQuery(fromArea),
    ...tokenizeAreaQuery(toArea)
  ];
  return {
    include: [...new Set([from, to, ...dynamic])].filter(Boolean),
    exclude: []
  };
}

function parseRouteFromAreaQuery(query?: string): {
  fromArea?: string;
  toArea?: string;
} {
  if (!query) return {};
  const match = query.match(
    /\bfrom\s+(.+?)\s+(?:to|towards?|heading to|on my way to)\s+(.+)$/i
  );
  if (!match) return {};
  return {
    fromArea: match[1].trim(),
    toArea: match[2].trim()
  };
}

function computeRouteScore(
  restroom: Restroom,
  routeProfile: RouteProfile
): number {
  if (routeProfile.include.length === 0) return 0;
  const haystack =
    `${restroom.name} ${restroom.address} ${restroom.city} ${restroom.notes}`.toLowerCase();

  let includeMatched = 0;
  for (const keyword of routeProfile.include) {
    if (!keyword || keyword.length < 2) continue;
    if (haystack.includes(keyword.toLowerCase())) includeMatched += 1;
  }

  let excludeMatched = 0;
  for (const keyword of routeProfile.exclude) {
    if (!keyword || keyword.length < 2) continue;
    if (haystack.includes(keyword.toLowerCase())) excludeMatched += 1;
  }

  if (excludeMatched > 0) return -8;
  if (includeMatched === 0) return -4;
  if (includeMatched >= 2) return 7;
  return 3;
}

type GeoConstraint = {
  label: string;
  locationTokens: string[];
  stateCode?: string;
  strictCity?: string;
};

type LandmarkCityHint = {
  strictCity: string;
  stateCode: string;
  tokens: string[];
};

const LANDMARK_CITY_HINTS: LandmarkCityHint[] = [
  {
    strictCity: "san francisco",
    stateCode: "ca",
    tokens: [
      "san francisco",
      "sf",
      "fishermans wharf",
      "fisherman's wharf",
      "union square",
      "mission district",
      "soma",
      "financial district"
    ]
  },
  {
    strictCity: "new york",
    stateCode: "ny",
    tokens: [
      "new york",
      "nyc",
      "times square",
      "central park",
      "soho",
      "lower manhattan",
      "upper west side",
      "midtown",
      "brooklyn bridge"
    ]
  },
  {
    strictCity: "boston",
    stateCode: "ma",
    tokens: [
      "boston",
      "back bay",
      "fenway",
      "charlestown",
      "seaport",
      "beacon hill",
      "north end",
      "south station"
    ]
  }
];

const STATE_NAME_TO_CODE: Record<string, string> = {
  alabama: "al",
  alaska: "ak",
  arizona: "az",
  arkansas: "ar",
  california: "ca",
  colorado: "co",
  connecticut: "ct",
  delaware: "de",
  florida: "fl",
  georgia: "ga",
  hawaii: "hi",
  idaho: "id",
  illinois: "il",
  indiana: "in",
  iowa: "ia",
  kansas: "ks",
  kentucky: "ky",
  louisiana: "la",
  maine: "me",
  maryland: "md",
  massachusetts: "ma",
  michigan: "mi",
  minnesota: "mn",
  mississippi: "ms",
  missouri: "mo",
  montana: "mt",
  nebraska: "ne",
  nevada: "nv",
  "new hampshire": "nh",
  "new jersey": "nj",
  "new mexico": "nm",
  "new york": "ny",
  "north carolina": "nc",
  "north dakota": "nd",
  ohio: "oh",
  oklahoma: "ok",
  oregon: "or",
  pennsylvania: "pa",
  "rhode island": "ri",
  "south carolina": "sc",
  "south dakota": "sd",
  tennessee: "tn",
  texas: "tx",
  utah: "ut",
  vermont: "vt",
  virginia: "va",
  washington: "wa",
  "west virginia": "wv",
  wisconsin: "wi",
  wyoming: "wy",
  "district of columbia": "dc"
};

function detectStateCode(text: string): string | undefined {
  const normalized = normalizeLocation(text);
  for (const code of Object.values(STATE_NAME_TO_CODE)) {
    if (new RegExp(`\\b${code}\\b`).test(normalized)) return code;
  }
  for (const [stateName, code] of Object.entries(STATE_NAME_TO_CODE)) {
    if (normalized.includes(stateName)) return code;
  }
  return undefined;
}

function detectLandmarkCityHint(text: string): LandmarkCityHint | undefined {
  const normalized = normalizeLocation(text);
  return LANDMARK_CITY_HINTS.find((hint) =>
    hint.tokens.some((token) => normalized.includes(normalizeLocation(token)))
  );
}

function resolveGeoConstraint(
  filters: RestroomSearchFilters,
  extraContext?: string
): GeoConstraint | undefined {
  const parts = [
    filters.areaQuery,
    filters.fromArea,
    filters.toArea,
    extraContext
  ].filter((value): value is string => Boolean(value));
  if (parts.length === 0) return undefined;

  let stateCode: string | undefined;
  let strictCity: string | undefined;
  for (const part of parts) {
    stateCode = stateCode ?? detectStateCode(part);
    const normalized = normalizeLocation(part);
    const hint = detectLandmarkCityHint(part);
    if (!strictCity && hint) {
      strictCity = hint.strictCity;
      stateCode = stateCode ?? hint.stateCode;
    }
    if (!strictCity && /\bsf\b/.test(normalized)) strictCity = "san francisco";
  }

  const tokenSet = new Set<string>();
  for (const part of parts) {
    for (const token of tokenizeAreaQuery(part)) {
      if (!Object.values(STATE_NAME_TO_CODE).includes(token)) {
        tokenSet.add(token);
      }
    }
  }

  const locationTokens = [...tokenSet].filter(
    (token) => token.length >= 3 && !/\d/.test(token)
  );
  if (!stateCode && locationTokens.length === 0) return undefined;

  const label = [
    locationTokens.slice(0, 2).join("-") || "requested-area",
    stateCode
  ]
    .filter(Boolean)
    .join("-");
  return { label, locationTokens, stateCode, strictCity };
}

function getStateCodeFromAddress(address: string): string | undefined {
  const match = address.match(/,\s*([A-Z]{2})(?:\s|$|,)/);
  return match?.[1]?.toLowerCase();
}

function matchesGeoConstraint(
  restroom: Restroom,
  scope: GeoConstraint
): boolean {
  const stateCode = getStateCodeFromAddress(restroom.address) ?? "";
  if (scope.stateCode && stateCode !== scope.stateCode) return false;
  if (scope.strictCity) {
    const city = normalizeLocation(restroom.city);
    if (city !== scope.strictCity) return false;
  }

  const haystack =
    `${restroom.name} ${restroom.address} ${restroom.city} ${restroom.notes}`.toLowerCase();
  if (scope.locationTokens.length === 0) return true;

  let matched = 0;
  for (const token of scope.locationTokens) {
    if (haystack.includes(token.toLowerCase())) matched += 1;
  }
  const minMatches = scope.locationTokens.length <= 2 ? 1 : 2;
  return matched >= minMatches;
}

async function lookupOpenStreetMapPlace(
  placeQuery: string
): Promise<OSMSearchResult | null> {
  const url =
    "https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=" +
    encodeURIComponent(placeQuery);
  const response = await fetch(url, {
    headers: {
      "User-Agent": "bathroom-access-concierge/1.0"
    }
  });
  if (!response.ok) return null;
  const results = (await response.json()) as OSMSearchResult[];
  return results[0] ?? null;
}

function overpassElementSelector(
  osmType: string,
  osmId: number
): string | null {
  if (osmType === "node") return `node(${osmId});`;
  if (osmType === "way") return `way(${osmId});`;
  if (osmType === "relation") return `relation(${osmId});`;
  return null;
}

async function lookupPlaceFeatureData(input: {
  placeName: string;
  city?: string;
  stateOrCountry?: string;
}) {
  const query = [input.placeName, input.city, input.stateOrCountry]
    .filter(Boolean)
    .join(", ");
  const osm = await lookupOpenStreetMapPlace(query);
  if (!osm) {
    return {
      found: false,
      query,
      message: "No matching place found in OpenStreetMap."
    };
  }

  const selector = overpassElementSelector(osm.osm_type, osm.osm_id);
  if (!selector) {
    return {
      found: true,
      query,
      place: osm.display_name,
      coordinates: { lat: Number(osm.lat), lon: Number(osm.lon) },
      message: "Place found, but could not read feature tags for this OSM type."
    };
  }

  const overpassQuery = `[out:json][timeout:10];${selector}out tags center;`;
  const overpassResp = await fetch("https://overpass-api.de/api/interpreter", {
    method: "POST",
    headers: { "Content-Type": "text/plain;charset=UTF-8" },
    body: overpassQuery
  });

  if (!overpassResp.ok) {
    return {
      found: true,
      query,
      place: osm.display_name,
      coordinates: { lat: Number(osm.lat), lon: Number(osm.lon) },
      message:
        "Place found, but feature lookup service is temporarily unavailable."
    };
  }

  const overpassData = (await overpassResp.json()) as {
    elements?: Array<{ tags?: Record<string, string> }>;
  };
  const tags = overpassData.elements?.[0]?.tags ?? {};
  return {
    found: true,
    query,
    place: osm.display_name,
    coordinates: { lat: Number(osm.lat), lon: Number(osm.lon) },
    features: {
      opening_hours: tags.opening_hours ?? null,
      wheelchair: tags.wheelchair ?? null,
      toilets_wheelchair: tags["toilets:wheelchair"] ?? null,
      toilets_gender_segregated: tags["toilets:gender_segregated"] ?? null,
      changing_table:
        tags["changing_table"] ?? tags["baby_changing_table"] ?? null,
      access: tags.access ?? null,
      fee: tags.fee ?? null
    },
    source: "OpenStreetMap (Nominatim + Overpass)",
    dataWarning:
      "OSM tags may be missing or outdated. Verify with the venue directly when critical."
  };
}

function haversineKm(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  const toRad = (d: number) => (d * Math.PI) / 180;
  const R = 6371;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) *
      Math.cos(toRad(lat2)) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function distanceScore(distanceKm: number): number {
  if (distanceKm <= 0.3) return 10;
  if (distanceKm <= 0.7) return 8;
  if (distanceKm <= 1.5) return 6;
  if (distanceKm <= 3) return 4;
  if (distanceKm <= 6) return 2;
  return 0;
}

function extractLandmarkQuery(areaQuery?: string): string | undefined {
  if (!areaQuery) return undefined;
  const q = areaQuery.trim();
  const nearMatch = q.match(
    /\b(?:near|around|by|close to|closest to)\s+(.+)$/i
  );
  if (nearMatch) return nearMatch[1].trim();
  return q;
}

function normalizeLandmarkToken(token: string): string {
  return token.replace(/\d+(st|nd|rd|th)\b/gi, "").trim();
}

export class ChatAgent extends AIChatAgent<Env, ConciergeState> {
  initialState: ConciergeState = {
    preferences: DEFAULT_PREFERENCES
  };

  private getPreferences(): UserPreferences {
    return this.state?.preferences ?? DEFAULT_PREFERENCES;
  }

  private getLastLocationQuery(): string | undefined {
    return this.state?.lastLocation?.areaQuery;
  }

  private applyStatePatch(patch: Partial<ConciergeState>) {
    this.setState({
      preferences: this.getPreferences(),
      lastLocation: this.state?.lastLocation,
      ...patch
    });
  }

  onStart() {
    // Configure OAuth popup behavior for MCP servers that require authentication
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }

  @callable()
  async addServer(name: string, url: string, host: string) {
    return await this.addMcpServer(name, url, { callbackHost: host });
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  @callable()
  async resetPreferences() {
    this.applyStatePatch({
      preferences: DEFAULT_PREFERENCES,
      lastLocation: undefined
    });
    return { message: "Preferences reset.", preferences: DEFAULT_PREFERENCES };
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    try {
      const workersai = createWorkersAI({ binding: this.env.AI });

      const result = streamText({
        // Keep to a broadly available Workers AI chat model for reliability.
        model: workersai("@cf/zai-org/glm-4.7-flash", {
          sessionAffinity: this.sessionAffinity
        }),
        system: `You are Bathroom Access Concierge, a user-facing assistant that helps people find suitable restrooms.

Your priorities are:
1) Suggest restrooms that match the user's needs.
2) Remember accessibility preferences when users share them.
3) Explain each recommendation clearly with specific reasons.

Behavior rules:
- When the user asks for a restroom recommendation, call findRestrooms first.
- If the user asks for options "on the way" or "from X to Y", pass both fromArea and toArea to findRestrooms.
- When the user mentions a lasting preference (wheelchairAccessible, singleStall, openNow, changingTable), call setAccessibilityPreferences.
- You may call getAccessibilityPreferences to include remembered preferences in your response.
- Always explain why the recommendation fits the user.
- If no restroom satisfies every requirement, say what trade-offs exist and suggest the closest options.
- Prefer results from the requested area, but if matches are sparse, return the closest available options.
- Only describe facts that appear in tool output fields. Do not invent guarantees or details.
- If a requested feature is missing or false, explicitly say it is unavailable.
- Call lookupPlaceFeatures only when the user explicitly asks for hours/accessibility/feature verification.
- Use databaseDetails first when available. Only call lookupPlaceFeatures for fields that are missing in databaseDetails.
- Do not list feature checklists unless the user explicitly asked for those specific features in their current request.
- Mention ONLY restroom names/addresses that appear in the latest findRestrooms recommendations array.
- If findRestrooms returns no recommendations, say no results found and ask a follow-up location question.
- Do not suggest general neighborhoods, transit stops, or venues unless they are present in tool output.`,
        messages: pruneMessages({
          messages: await convertToModelMessages(this.messages),
          toolCalls: "before-last-2-messages"
        }),
        temperature: 0,
        tools: {
          setAccessibilityPreferences: tool({
            description:
              "Save or update the user's restroom accessibility preferences for future recommendations.",
            inputSchema: z.object({
              wheelchairAccessible: z
                .boolean()
                .optional()
                .describe(
                  "Whether wheelchair-accessible restrooms are required."
                ),
              singleStall: z
                .boolean()
                .optional()
                .describe("Whether single-stall restrooms are preferred."),
              openNow: z
                .boolean()
                .optional()
                .describe("Whether the restroom must currently be open."),
              changingTable: z
                .boolean()
                .optional()
                .describe("Whether a changing table is required.")
            }),
            execute: async (update) => {
              const current = this.getPreferences();
              const next = mergePreferences(current, update);
              this.applyStatePatch({ preferences: next });
              return {
                message: "Preferences saved.",
                preferences: next
              };
            }
          }),

          getAccessibilityPreferences: tool({
            description:
              "Retrieve the current saved restroom accessibility preferences.",
            inputSchema: z.object({}),
            execute: async () => {
              return {
                preferences: this.getPreferences()
              };
            }
          }),

          lookupPlaceFeatures: tool({
            description:
              "Look up a specific place and fetch available tags such as opening hours and wheelchair-related fields from OpenStreetMap.",
            inputSchema: z.object({
              placeName: z
                .string()
                .describe("Specific place name, e.g. Boston Public Library."),
              city: z.string().optional().describe("Optional city filter."),
              stateOrCountry: z
                .string()
                .optional()
                .describe("Optional state, region, or country.")
            }),
            execute: async ({ placeName, city, stateOrCountry }) => {
              const latestUserText = extractLatestUserText(this.messages);
              if (!userExplicitlyAskedForFeatureLookup(latestUserText)) {
                return {
                  found: false,
                  skipped: true,
                  reason:
                    "User did not explicitly ask for hours/accessibility feature lookup."
                };
              }
              try {
                return await lookupPlaceFeatureData({
                  placeName,
                  city,
                  stateOrCountry
                });
              } catch (error) {
                return {
                  found: false,
                  query: [placeName, city, stateOrCountry]
                    .filter(Boolean)
                    .join(", "),
                  message: `Lookup failed: ${error}`
                };
              }
            }
          }),

          findRestrooms: tool({
            description:
              "Find restroom recommendations based on remembered preferences and optional request filters.",
            inputSchema: z.object({
              areaQuery: z
                .string()
                .optional()
                .describe(
                  "Area hint like lobby, library, food court, or transit."
                ),
              fromArea: z
                .string()
                .optional()
                .describe(
                  "Route start area. Use when user asks for options along a journey."
                ),
              toArea: z
                .string()
                .optional()
                .describe(
                  "Route destination area. Use with fromArea for along-the-way suggestions."
                ),
              wheelchairAccessible: z.boolean().optional(),
              singleStall: z.boolean().optional(),
              openNow: z.boolean().optional(),
              changingTable: z.boolean().optional()
            }),
            execute: async (filters) => {
              const prefs = this.getPreferences();
              const typedFilters: RestroomSearchFilters = filters;
              const latestUserText = extractLatestUserText(this.messages);
              const priorLocationQuery = this.getLastLocationQuery();
              const usePriorLocation =
                !typedFilters.areaQuery &&
                isLikelyFollowUpFeatureQuery(latestUserText) &&
                Boolean(priorLocationQuery);
              const inferredAreaQuery =
                typedFilters.areaQuery?.trim() ||
                (usePriorLocation ? priorLocationQuery : latestUserText);
              const normalizedFilters: RestroomSearchFilters = {
                ...typedFilters,
                areaQuery: inferredAreaQuery
              };
              const effective: UserPreferences = {
                wheelchairAccessible:
                  normalizedFilters.wheelchairAccessible ??
                  prefs.wheelchairAccessible,
                singleStall: normalizedFilters.singleStall ?? prefs.singleStall,
                openNow: normalizedFilters.openNow ?? prefs.openNow,
                changingTable:
                  normalizedFilters.changingTable ?? prefs.changingTable
              };
              const requestedForDisplay = {
                wheelchairAccessible:
                  normalizedFilters.wheelchairAccessible === true,
                singleStall: normalizedFilters.singleStall === true,
                openNow: normalizedFilters.openNow === true,
                changingTable: normalizedFilters.changingTable === true
              };
              const parsedRoute = parseRouteFromAreaQuery(
                normalizedFilters.areaQuery
              );
              const fromArea =
                normalizedFilters.fromArea ?? parsedRoute.fromArea;
              const toArea = normalizedFilters.toArea ?? parsedRoute.toArea;
              const hasRouteContext = Boolean(fromArea && toArea);
              const routeProfile = findRouteProfile(fromArea, toArea);
              const geoScope = resolveGeoConstraint(
                {
                  ...normalizedFilters,
                  fromArea,
                  toArea
                },
                latestUserText
              );
              let scoped = hasRouteContext
                ? RESTROOMS
                : RESTROOMS.filter((restroom) => {
                    if (!normalizedFilters.areaQuery) return true;
                    const fullQuery = normalizedFilters.areaQuery.toLowerCase();
                    const tokens = tokenizeAreaQuery(
                      normalizedFilters.areaQuery
                    );
                    const haystack =
                      `${restroom.name} ${restroom.address} ${restroom.city}`.toLowerCase();

                    if (haystack.includes(fullQuery)) return true;
                    if (tokens.length === 0) return false;

                    let tokenMatches = 0;
                    for (const token of tokens) {
                      if (haystack.includes(token)) tokenMatches += 1;
                    }

                    // For short place hints ("downtown boston"), allow 1 match.
                    // Keep this permissive so user can refine after seeing results.
                    const minMatches = 1;
                    return tokenMatches >= minMatches;
                  });

              // If the user provided a single city token (e.g. "denver"), prefer exact city matches.
              if (!hasRouteContext && normalizedFilters.areaQuery) {
                const tokens = tokenizeAreaQuery(normalizedFilters.areaQuery);
                if (tokens.length === 1) {
                  const cityExact = scoped.filter(
                    (restroom) => normalizeLocation(restroom.city) === tokens[0]
                  );
                  if (cityExact.length > 0) {
                    scoped = cityExact;
                  }
                }
              }
              const geoScoped = geoScope
                ? scoped.filter((restroom) =>
                    matchesGeoConstraint(restroom, geoScope)
                  )
                : scoped;
              const geoConstrainedPool =
                geoScope && geoScoped.length === 0 && !usePriorLocation
                  ? scoped
                  : geoScoped;

              if (usePriorLocation && geoConstrainedPool.length === 0) {
                return {
                  rememberedPreferences: prefs,
                  filtersApplied: {
                    ...normalizedFilters,
                    fromArea,
                    toArea,
                    ...effective
                  },
                  strictMatchFound: false,
                  recommendations: [],
                  totalMatches: 0
                };
              }

              const matching = geoConstrainedPool.filter((restroom) =>
                isMatchingRequirements(restroom, effective)
              );
              const candidatePool =
                matching.length > 0 ? matching : geoConstrainedPool;
              const scored = candidatePool.map((restroom) => {
                const { score, reasons } = scoreRestroom(restroom, effective);
                const routeScore = computeRouteScore(restroom, routeProfile);
                const areaScore = computeAreaRelevance(
                  restroom,
                  normalizedFilters.areaQuery
                );
                const routeReasons =
                  routeProfile.include.length > 0 && routeScore > 0
                    ? ["matches route corridor"]
                    : [];
                return {
                  restroom,
                  score: score + routeScore + areaScore,
                  reasons: [...reasons, ...routeReasons],
                  routeScore
                };
              });

              const routeMatchedCount = scored.filter(
                (entry) => entry.routeScore > 0
              ).length;
              let ranked = (
                routeMatchedCount > 0
                  ? scored.filter((entry) => entry.routeScore > 0)
                  : scored
              ).sort((a, b) => b.score - a.score);

              // Landmark-aware reranking: if user mentions a place/area,
              // geocode it and prioritize restrooms by actual proximity.
              const landmarkQuery = extractLandmarkQuery(
                normalizedFilters.areaQuery
              );
              if (landmarkQuery) {
                const landmarkHint = normalizeLandmarkToken(landmarkQuery);
                const landmarkLookupQuery = [
                  landmarkHint,
                  geoScope?.strictCity,
                  geoScope?.stateCode?.toUpperCase()
                ]
                  .filter(Boolean)
                  .join(", ");
                const landmark =
                  await lookupOpenStreetMapPlace(landmarkLookupQuery);
                if (landmark) {
                  const landmarkLat = Number(landmark.lat);
                  const landmarkLon = Number(landmark.lon);
                  const geoCache = new Map<
                    string,
                    { lat: number; lon: number } | null
                  >();

                  async function getRestroomCoords(restroom: Restroom) {
                    const key = `${restroom.name}|${restroom.address}|${restroom.city}`;
                    if (geoCache.has(key)) return geoCache.get(key) ?? null;
                    const q = `${restroom.name}, ${restroom.address}, ${restroom.city}`;
                    const r = await lookupOpenStreetMapPlace(q);
                    if (!r) {
                      geoCache.set(key, null);
                      return null;
                    }
                    const coords = { lat: Number(r.lat), lon: Number(r.lon) };
                    geoCache.set(key, coords);
                    return coords;
                  }

                  const topForDistance = ranked.slice(0, 40);
                  const withDistance = await Promise.all(
                    topForDistance.map(async (entry) => {
                      const coords = await getRestroomCoords(entry.restroom);
                      if (!coords)
                        return { ...entry, distanceKm: null, dScore: 0 };
                      const km = haversineKm(
                        landmarkLat,
                        landmarkLon,
                        coords.lat,
                        coords.lon
                      );
                      const dScore = distanceScore(km);
                      return { ...entry, distanceKm: km, dScore };
                    })
                  );

                  const nearbyOnly = withDistance.filter(
                    (entry) =>
                      entry.distanceKm !== null && entry.distanceKm <= 12
                  );
                  const distancePool =
                    nearbyOnly.length > 0 ? nearbyOnly : withDistance;

                  const rerankedTop = distancePool
                    .map((entry) => ({
                      ...entry,
                      score: entry.score + entry.dScore,
                      reasons:
                        entry.distanceKm !== null
                          ? [
                              ...entry.reasons,
                              `near ${landmarkHint} (${entry.distanceKm.toFixed(1)} km)`
                            ]
                          : entry.reasons
                    }))
                    .sort((a, b) => b.score - a.score);

                  ranked = [...rerankedTop, ...ranked.slice(40)];
                }
              }

              const locationTokens = tokenizeAreaQuery(
                normalizedFilters.areaQuery ?? ""
              );
              const looksLikeLocation =
                locationTokens.length > 0 ||
                Boolean(
                  resolveGeoConstraint(normalizedFilters, latestUserText)
                );
              if (looksLikeLocation && normalizedFilters.areaQuery) {
                this.applyStatePatch({
                  lastLocation: { areaQuery: normalizedFilters.areaQuery }
                });
              }

              return {
                rememberedPreferences: prefs,
                filtersApplied: {
                  ...normalizedFilters,
                  fromArea,
                  toArea,
                  ...effective
                },
                strictMatchFound: matching.length > 0,
                recommendations: ranked.slice(0, 3).map((r) => {
                  const formatted = formatRestroom(r.restroom, r.reasons);
                  return {
                    ...formatted,
                    databaseDetails: getDatabaseDetails(r.restroom),
                    features: pickRequestedFeatures(
                      formatted.features as {
                        wheelchairAccessible: boolean;
                        singleStall: boolean;
                        openNow: boolean;
                        changingTable: boolean;
                      },
                      requestedForDisplay
                    )
                  };
                }),
                totalMatches: candidatePool.length
              };
            }
          })
        },
        stopWhen: stepCountIs(5),
        abortSignal: options?.abortSignal
      });

      return result.toUIMessageStreamResponse();
    } catch (error) {
      console.error("ChatAgent onChatMessage failed:", error);
      const stream = createUIMessageStream({
        execute: ({ writer }) => {
          const partId = "fallback-error-text";
          writer.write({ type: "text-start", id: partId });
          writer.write({
            type: "text-delta",
            id: partId,
            delta:
              "I hit an internal issue while searching restrooms. Please try once more. "
          });
          writer.write({
            type: "text-delta",
            id: partId,
            delta:
              "If this keeps happening, tell me your city and I will run a simpler local search."
          });
          writer.write({ type: "text-end", id: partId });
        }
      });
      return createUIMessageStreamResponse({ stream });
    }
  }
}

export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
