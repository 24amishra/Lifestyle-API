import { NextResponse } from "next/server";

export async function GET() {
  // NEXT_PUBLIC_ vars are only guaranteed on the client side.
  // For server-side Route Handlers, prefer a non-prefixed env var first.
  const API_BASE_URL =
    process.env.API_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://localhost:5000";
  return NextResponse.redirect(`${API_BASE_URL}/authorize`);
}
