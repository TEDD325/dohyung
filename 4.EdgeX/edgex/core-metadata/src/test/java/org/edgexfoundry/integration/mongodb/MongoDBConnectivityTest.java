/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.integration.mongodb;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.fail;

import java.net.UnknownHostException;

import org.edgexfoundry.test.category.RequiresMongoDB;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import com.mongodb.DB;
import com.mongodb.DBCollection;
import com.mongodb.MongoClient;
import com.mongodb.MongoClientURI;
import com.mongodb.MongoTimeoutException;

@Category(RequiresMongoDB.class)
public class MongoDBConnectivityTest {

  private static final String MONGO_URI = "mongodb://meta:password@localhost/metadata";
  private static final String DB_NAME = "metadata";
  // checking for a sampling of collections
  private static final String DEVICE_COLLECTION_NAME = "Device";
  private static final String DEVICE_SERVICE_COLLECTION_NAME = "DeviceService";
  private static final String DEVICE_PROFILE_COLLECTION_NAME = "DeviceProfile";

  @Test
  public void testMongoDBConnect() throws UnknownHostException {
    MongoClient mongoClient = new MongoClient(new MongoClientURI(MONGO_URI));
    DB database = mongoClient.getDB(DB_NAME);
    String[] collectionNames =
        {DEVICE_COLLECTION_NAME, DEVICE_PROFILE_COLLECTION_NAME, DEVICE_SERVICE_COLLECTION_NAME};
    for (String name : collectionNames) {
      DBCollection collection = database.getCollection(name);
      try {
        assertFalse("MongoDB collection " + name + " not accessible", collection.isCapped());
      } catch (MongoTimeoutException ex) {
        fail("Mongo DB not available.  Check that Mongo DB has been started");
      }
    }
  }

}
