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

package org.edgexfoundry.dao.integration;

import static org.edgexfoundry.test.data.ServiceData.TEST_SERVICE_NAME;
import static org.edgexfoundry.test.data.ServiceData.checkTestData;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

import org.edgexfoundry.Application;
import org.edgexfoundry.dao.AddressableRepository;
import org.edgexfoundry.dao.DeviceServiceDao;
import org.edgexfoundry.dao.DeviceServiceRepository;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.edgexfoundry.test.data.AddressableData;
import org.edgexfoundry.test.data.ServiceData;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({RequiresMongoDB.class, RequiresSpring.class, RequiresWeb.class})
public class DeviceServiceDaoTest {

  @Autowired
  private DeviceServiceDao dao;

  @Autowired
  private DeviceServiceRepository repos;

  @Autowired
  private AddressableRepository addrRepos;

  private String id;

  @Before
  public void createTestData() {
    Addressable addr = AddressableData.newTestInstance();
    addrRepos.save(addr);
    DeviceService service = ServiceData.newTestInstance();
    service.setAddressable(addr);
    repos.save(service);
    id = service.getId();
    assertNotNull("new test Device Service has no identifier", id);
    assertNotNull("Dao is null", dao);
  }

  @After
  public void cleanup() {
    repos.deleteAll();
    addrRepos.deleteAll();
  }

  @Test
  public void testGetById() {
    DeviceService service = dao.getById(id);
    assertNotNull("Device Service is null on getById", service);
    checkTestData(service, id);
  }

  @Test
  public void testGetByIdWithBadId() {
    assertNull("No device service should be found with bad id", dao.getById("badid"));
  }

  @Test
  public void testGetByName() {
    DeviceService service = dao.getByName(TEST_SERVICE_NAME);
    assertNotNull("Device service is null on getByName", service);
    checkTestData(service, id);
  }

  @Test
  public void testGetByNameWithBadName() {
    assertNull("No device service should be found with bad name", dao.getByName("badname"));
  }

  @Test
  public void testGetByIdOrName() {
    DeviceService service = new DeviceService();
    service.setName(TEST_SERVICE_NAME);
    assertNotNull("Device service is null on getByIdOrName with valid name", dao.getByIdOrName(service));
    service.setName(null);
    service.setId(id);
    assertNotNull("Device service is null on getByIdOrName with valid id", dao.getByIdOrName(service));
  }

  @Test
  public void testGetByIdOrNameWithNull() {
    assertNull("No device service should be found with null on getByIdOrName",
        dao.getByIdOrName(null));
  }

  @Test
  public void testGetByIdOrNameWithBadIdentifiers() {
    DeviceService service = new DeviceService();
    service.setId("badid");
    assertNull("No device service should be found with bad id on getByIdOrName",
        dao.getByIdOrName(service));
    service.setId(null);
    service.setName("badname");
    assertNull("No device service should be found with bad name on getByIdOrName",
        dao.getByIdOrName(service));
  }

}
