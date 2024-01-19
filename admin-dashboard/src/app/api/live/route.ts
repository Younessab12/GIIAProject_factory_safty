import { NextRequest, NextResponse } from "next/server";
import { db } from "~/server/db";

export async function GET(request: NextRequest) {
  try{
    // get active operators with their status

    await db.operator.updateMany({
      where: {
        lastUpdated: {
          lte: new Date(Date.now() - 60*1000),
        },
      },
      data: {
        status: "inactive",
      },
    });

    const operators = await db.operator.findMany({
      select: { operatorName: true, status: true, lastUpdated: true },
    });

    return new NextResponse(
      JSON.stringify({
        operators,
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