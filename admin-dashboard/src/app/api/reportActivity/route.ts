import { NextRequest, NextResponse } from "next/server";
import { db } from "~/server/db";


type Activity = {
  apiKey: string;
  activityName: string;
  gravity: number;
  timestamp: string;
  operatorName: string;
}

export async function POST(request: NextRequest) {
  // const { nameLookup }: any = await request.json();
  const activity : Activity = await request.json();
  

  activity.timestamp = new Date(activity.timestamp);
  console.log(activity);
  
  if (!activity?.apiKey || !activity.activityName || !activity.gravity || !activity.timestamp || !activity.operatorName) {
    return new NextResponse(
      JSON.stringify({ name: "Please provide all required fields" }),
      { status: 400 }
    );
  }

  try {const raspberry = await db.raspberry.findUnique({
    where: { apiKey: activity.apiKey },
  });

  if(!raspberry) {
    return new NextResponse(
      JSON.stringify({ name: "API not valid" }),
      { status: 404 }
    );
  }

  let operator = await db.operator.findUnique({
    where: { operatorName: activity.operatorName },
  });
  console.log(operator)
  if(!operator) {
    operator = await db.operator.create({
      data: {
        operatorName: activity.operatorName,
      },
    });
  }

  const newActivity = await db.activity.create({
    data: {
      activityName: activity.activityName,
      gravity: activity.gravity,
      timestamp: new Date (activity.timestamp),
      operator: {
        connect: {
          id: operator.id,
        },
      },
      raspberry: {
        connect: {
          id: raspberry.id,
        },
      },
    },
  });

  await db.operator.update({
    where: { operatorName: activity.operatorName },
    data: {
      status: "danger",
      lastUpdated: new Date(),
    },
  });

  return new NextResponse(JSON.stringify(newActivity));
  } catch (error) {
    return new NextResponse(
      JSON.stringify({ name: "Something went wrong" + error.message}),
      { status: 500 }
    );
  }
  
  
}