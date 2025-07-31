import { NextResponse } from 'next/server';

export async function GET() {

  const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://first-production-254a.up.railway.app'
  : 'http://localhost:5000';
  return NextResponse.redirect(`${API_BASE_URL}/authorize`);
}