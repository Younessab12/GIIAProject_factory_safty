import { NextRequest, NextResponse } from "next/server";
import { db } from "~/server/db";

export async function GET(request: NextRequest) {
  try{
    // get paramater {operatorName}
    //get last 20 operator
    const operatorName = request.nextUrl.searchParams.get("operatorName");
    const activityToFetch = request.nextUrl.searchParams.get("activities");

    if(!operatorName) {
      return new NextResponse(
        JSON.stringify({ name: "Provide operator" }),
        { status: 404 }
      );
    }

    const operator = await db.operator.findUnique({
      where: { operatorName: operatorName},
    });

    if(!operator) {
      return new NextResponse(
        JSON.stringify({ name: "Operator not found" }),
        { status: 404 }
      );
    }

    const data = {
      operator: operator,
      activities: []
    }

    try{
      const activities = await db.activity.findMany({
        take: activityToFetch ? parseInt(activityToFetch) : 20,
        orderBy: {
          timestamp: "desc",
        },
        where: {
          operatorId: operator.id
        }
      });

      data.activities = activities;
    } catch (err) {
      console.log(err);
    }

    return new NextResponse(
      JSON.stringify(data),
      { status: 200 }
    );

    
  } catch (err) {
    return new NextResponse(
      JSON.stringify({ name: "Error" }),
      { status: 400 }
    );
  }
}