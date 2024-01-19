import { NextRequest, NextResponse } from "next/server";
import { db } from "~/server/db";

type Activity = {
  apiKey: string;
  operatorName: string;
}

export async function POST(request: NextRequest) {
  // const { nameLookup }: any = await request.json();
  const activity : Activity = await request.json();
  
  if (!activity?.apiKey || !activity.operatorName) {
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

  if(!operator) {
    operator = await db.operator.create({
      data: {
        operatorName: activity.operatorName,
      },
    });
  }
  if( operator.status != "active" && operator.status != "inactive" && operator.lastUpdated != null && operator.lastUpdated > new Date(Date.now() - 60*1000)) 
  {
    return new NextResponse(
      JSON.stringify({ name: "Operator is already active" }),
      { status: 400 }
    );
  }
  operator = await db.operator.update({
    where: { operatorName: activity.operatorName },
    data: {
      status: "active",
      lastUpdated: new Date(),
    },
  });

  return new NextResponse(
    JSON.stringify({
      "lastRefresh": new Date().toISOString(),
    }),
    { status: 200 }
  );
  } catch (error) {
    return new NextResponse(
      JSON.stringify({ name: "Something went wrong" + error.message}),
      { status: 500 }
    );
  }
  
  
}