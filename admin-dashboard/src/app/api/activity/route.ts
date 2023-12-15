import { NextRequest, NextResponse } from "next/server";
import { db } from "~/server/db";


export async function GET(request: NextRequest) {
  try{
    //get last 20 activities
    const activityId = request.nextUrl.searchParams.get("activityId");

    if(!activityId) {
      return new NextResponse(
        JSON.stringify({ name: "Provide activity" }),
        { status: 404 }
      );
    }

    const activity = await db.activity.findUnique({
      where: { activityId: activityId},
      include: {
        operator: true,
      },
    });

    
    return new NextResponse(
      JSON.stringify(activity),
      { status: 200 }
    );
  } catch (err) {
    return new NextResponse(
      JSON.stringify({ name: "Error" }),
      { status: 400 }
    );
  }
}