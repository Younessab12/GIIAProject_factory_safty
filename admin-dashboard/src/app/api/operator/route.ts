import { NextRequest, NextResponse } from "next/server";
import { db } from "~/server/db";

export async function GET(request: NextRequest) {
  try{
    // get paramater {operatorName}
    //get last 20 operator
    const { operatorName: string } = request.query;

    const operator = await db.operator.findUnique({
      where: { operatorName: operatorName },
    });

    if(!operator) {
      return new NextResponse(
        JSON.stringify({ name: "Operator not found" }),
        { status: 404 }
      );
    }



    

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