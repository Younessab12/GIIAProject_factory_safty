import { NextRequest, NextResponse } from "next/server";
import { db } from "~/server/db";


type Activity = {
  timestamp: string,
  operatorId: string,
  activityName: string,
  gravity: string
}

export async function GET(request: NextRequest) {
  try{
    //get last 20 activities
    const lastActivities = await db.activity.findMany({
      take: 20,
      orderBy: {
        timestamp: "desc",
      },
      include: {
        operator: true,
      },
    });

    return new NextResponse(
      JSON.stringify({
        "rows": lastActivities,
        "lastRefresh": new Date().toISOString(),
      }),
      { status: 200 }
    );
  } catch (err) {
    return new NextResponse(
      JSON.stringify({ name: "Error" }),
      { status: 400 }
    );
  }
}